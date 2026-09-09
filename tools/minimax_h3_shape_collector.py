#!/usr/bin/env python3
"""Static GEMM workload collector for FastVideo MiniMax H3 SFT.

This script derives the GEMM shapes that the MiniMax H3 training path presents
per sequence-parallel rank, without materializing the 33B model weights.

It is intended for the FastVideo MiniMax H3 T2VA SFT configuration.  The
collector:
  * reads the training geometry/SP size from the experiment YAML;
  * obtains the valid text-token count from a preprocessed parquet row or from
    --text-tokens;
  * reconstructs packed text/audio/video sequence length;
  * applies the same "pad then shard" rule used by FastVideo sequence parallel;
  * emits logical Fwd/Dgrad/Wgrad GEMM workloads for Q/K/V/O, FFN Gate+Up/Down,
    block AdaLN, final AdaLN, and (optionally) the two text-refiner blocks.

The output is a *logical/static* workload description. Runtime stride,
contiguity, actual cuBLAS/cuBLASLt kernel choice, and checkpoint-recompute call
counts must be verified by runtime instrumentation/profiling later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


# FastVideo MiniMax H3 architecture defaults at commit
# 556ac7088e7b4750806d277d31e0db6cd25a5238.
DEFAULT_HIDDEN_SIZE = 5376
DEFAULT_NUM_HEADS = 56
DEFAULT_HEAD_DIM = 128
DEFAULT_FFN_DIM = 14336
DEFAULT_NUM_LAYERS = 50
DEFAULT_NUM_REFINER_LAYERS = 2
DEFAULT_TIME_EMBED_DIM = 2688
DEFAULT_MODALITIES = 3
DEFAULT_PATCH_SIZE = (1, 2, 2)

# FastVideo MiniMax H3 packing/training constants.
DEFAULT_FPS = 24
DEFAULT_AUDIO_LATENTS_PER_SECOND = 40
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_VAE_SPATIAL_DOWNSAMPLE = 16


@dataclass(frozen=True)
class Arch:
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_heads: int = DEFAULT_NUM_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    ffn_dim: int = DEFAULT_FFN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    num_refiner_layers: int = DEFAULT_NUM_REFINER_LAYERS
    time_embed_dim: int = DEFAULT_TIME_EMBED_DIM
    modalities: int = DEFAULT_MODALITIES
    patch_t: int = DEFAULT_PATCH_SIZE[0]
    patch_h: int = DEFAULT_PATCH_SIZE[1]
    patch_w: int = DEFAULT_PATCH_SIZE[2]

    @property
    def attention_inner_dim(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class Geometry:
    text_tokens: int
    video_rows: int
    audio_rows: int
    global_sequence: int
    padded_global_sequence: int
    main_local_rows: int
    refiner_local_rows: int
    latent_height: int
    latent_width: int
    audio_latents: int
    sp_size: int


@dataclass
class Workload:
    scope: str
    module: str
    module_pattern: str
    phase: str
    token_rows: int
    input_features: int
    output_features: int
    gemm_m: int
    gemm_n: int
    gemm_k: int
    dtype: str
    logical_count_per_step: int
    full_model_reference_count: int
    flops_per_call: int
    weighted_flops_per_step: int
    m_mod_16: int
    n_mod_16: int
    k_mod_16: int
    m_mod_128: int
    n_mod_128: int
    k_mod_128: int
    equation: str
    static_input_stride: str
    static_weight_stride: str
    runtime_stride_required: bool
    source: str
    notes: str = ""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return data


def nested_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _flatten_scalars(value: Any) -> Iterable[Any]:
    """Recursively flatten pyarrow/list/numpy-ish nested mask values."""
    if value is None:
        return
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_scalars(item)
    else:
        yield value


def text_tokens_from_parquet(path: Path, row: int, column: str) -> tuple[int, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyarrow is required for --parquet: pip install pyarrow") from exc

    if column not in pq.read_schema(path).names:
        # T2VA single-sample rows store no mask column; the text embedding is
        # saved unpadded, so its leading dimension is the valid token count.
        fallback_column = "text_embedding_shape"
        shape_table = pq.read_table(path, columns=[fallback_column])
        shape = shape_table[fallback_column][row].as_py()
        if not shape:
            raise ValueError(f"{fallback_column!r} in row {row} is empty")
        return int(shape[0]), fallback_column + "[0]"
    table = pq.read_table(path, columns=[column])
    value = table[column][row].as_py()
    flattened = list(_flatten_scalars(value))
    if not flattened:
        raise ValueError(f"{column!r} in row {row} is empty")
    count = sum(bool(x) for x in flattened)
    if count <= 0:
        raise ValueError(f"{column!r} in row {row} contains no valid text tokens")
    return int(count), column


def resolve_git_revision(start: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_geometry(
    *,
    text_tokens: int,
    num_latent_t: int,
    num_frames: int,
    height: int,
    width: int,
    sp_size: int,
    arch: Arch,
    vae_spatial_downsample: int,
    fps: int,
    audio_latents_per_second: int,
    audio_channels: int,
) -> Geometry:
    if min(text_tokens, num_latent_t, num_frames, height, width, sp_size) <= 0:
        raise ValueError("text tokens, geometry, and SP size must all be positive")
    if height % vae_spatial_downsample or width % vae_spatial_downsample:
        raise ValueError(
            f"height/width ({height}x{width}) must be divisible by VAE downsample "
            f"factor {vae_spatial_downsample}"
        )

    latent_h = height // vae_spatial_downsample
    latent_w = width // vae_spatial_downsample
    if num_latent_t % arch.patch_t or latent_h % arch.patch_h or latent_w % arch.patch_w:
        raise ValueError(
            "latent geometry is not divisible by DiT patch size: "
            f"({num_latent_t}, {latent_h}, {latent_w}) vs "
            f"({arch.patch_t}, {arch.patch_h}, {arch.patch_w})"
        )

    video_rows = (
        (num_latent_t // arch.patch_t)
        * (latent_h // arch.patch_h)
        * (latent_w // arch.patch_w)
    )
    # Match FastVideo: int(round(num_frames / FPS * audio_latents_per_second)).
    audio_latents = int(round(num_frames / fps * audio_latents_per_second))
    audio_rows = audio_latents * audio_channels

    global_sequence = text_tokens + video_rows + audio_rows
    padded_global_sequence = math.ceil(global_sequence / sp_size) * sp_size
    main_local_rows = padded_global_sequence // sp_size

    # _refined_text() sequence-shards the text stream separately before the
    # two token-refiner blocks.
    refiner_local_rows = math.ceil(text_tokens / sp_size)

    return Geometry(
        text_tokens=text_tokens,
        video_rows=video_rows,
        audio_rows=audio_rows,
        global_sequence=global_sequence,
        padded_global_sequence=padded_global_sequence,
        main_local_rows=main_local_rows,
        refiner_local_rows=refiner_local_rows,
        latent_height=latent_h,
        latent_width=latent_w,
        audio_latents=audio_latents,
        sp_size=sp_size,
    )


def add_linear_triplet(
    out: list[Workload],
    *,
    scope: str,
    module: str,
    module_pattern: str,
    rows: int,
    in_features: int,
    out_features: int,
    dtype: str,
    count: int,
    full_model_reference_count: int | None = None,
    source: str,
    notes: str = "",
) -> None:
    """Add logical forward, input-gradient, and weight-gradient GEMMs.

    Canonical GEMM notation here is C[m,n] = A[m,k] @ B[k,n].
    PyTorch F.linear stores W as [out_features, in_features].

    Fwd:   Y  = X @ W^T       => (m=rows,        n=out, k=in)
    Dgrad: dX = dY @ W         => (m=rows,        n=in,  k=out)
    Wgrad: dW = dY^T @ X       => (m=out_features,n=in,  k=rows)

    Backend libraries may represent/transposed-dispatch the same logical GEMM
    differently; runtime profiler data is authoritative for actual kernel args.
    """
    base = dict(
        scope=scope,
        module=module,
        module_pattern=module_pattern,
        token_rows=rows,
        input_features=in_features,
        output_features=out_features,
        dtype=dtype,
        logical_count_per_step=count,
        full_model_reference_count=(
            50 if scope == "main_transformer" else count
            if full_model_reference_count is None else full_model_reference_count),
        static_input_stride=f"({in_features}, 1) [assumed contiguous for microbench only]",
        static_weight_stride=f"({in_features}, 1) for stored W[{out_features},{in_features}]",
        runtime_stride_required=True,
        source=source,
        notes=notes,
    )
    def make(phase: str, m: int, n: int, k: int, equation: str) -> Workload:
        flops = 2 * m * n * k
        return Workload(
            **base,
            phase=phase,
            gemm_m=m,
            gemm_n=n,
            gemm_k=k,
            flops_per_call=flops,
            weighted_flops_per_step=flops * count,
            m_mod_16=m % 16,
            n_mod_16=n % 16,
            k_mod_16=k % 16,
            m_mod_128=m % 128,
            n_mod_128=n % 128,
            k_mod_128=k % 128,
            equation=equation,
        )

    out.extend(
        [
            make(
                "Fwd", rows, out_features, in_features,
                f"[{rows},{in_features}] @ [{in_features},{out_features}] -> [{rows},{out_features}]",
            ),
            make(
                "Dgrad", rows, in_features, out_features,
                f"[{rows},{out_features}] @ [{out_features},{in_features}] -> [{rows},{in_features}]",
            ),
            make(
                "Wgrad", out_features, in_features, rows,
                f"[{out_features},{rows}] @ [{rows},{in_features}] -> [{out_features},{in_features}]",
            ),
        ]
    )


def build_workloads(
    geometry: Geometry,
    arch: Arch,
    dtype: str,
    include_refiner: bool,
    adaln_unique_timesteps: int,
) -> list[Workload]:
    workloads: list[Workload] = []
    inner = arch.attention_inner_dim
    main_rows = geometry.main_local_rows
    src = "FastVideo MiniMax H3 static derivation"

    for name in ("Q", "K", "V"):
        add_linear_triplet(
            workloads,
            scope="main_transformer",
            module=name,
            module_pattern=f"transformer_blocks.*.attn.to_{name.lower()}",
            rows=main_rows,
            in_features=arch.hidden_size,
            out_features=inner,
            dtype=dtype,
            count=arch.num_layers,
            source=src,
            notes="M is packed text+audio+video rows after SP pad/shard.",
        )

    add_linear_triplet(
        workloads,
        scope="main_transformer",
        module="O",
        module_pattern="transformer_blocks.*.attn.to_out",
        rows=main_rows,
        in_features=inner,
        out_features=arch.hidden_size,
        dtype=dtype,
        count=arch.num_layers,
        source=src,
    )
    add_linear_triplet(
        workloads,
        scope="main_transformer",
        module="Gate+Up",
        module_pattern="transformer_blocks.*.ff.fc_in",
        rows=main_rows,
        in_features=arch.hidden_size,
        out_features=2 * arch.ffn_dim,
        dtype=dtype,
        count=arch.num_layers,
        source=src,
        notes="One fused fc_in projection; output is chunked into value/gate halves.",
    )
    add_linear_triplet(
        workloads,
        scope="main_transformer",
        module="Down",
        module_pattern="transformer_blocks.*.ff.fc_out",
        rows=main_rows,
        in_features=arch.ffn_dim,
        out_features=arch.hidden_size,
        dtype=dtype,
        count=arch.num_layers,
        source=src,
    )

    # In T2VA SFT, build_row_timesteps normally yields two unique values:
    # one video/text timestep and one audio timestep. This M does NOT scale
    # with packed sequence length or SP size.
    add_linear_triplet(
        workloads,
        scope="main_transformer",
        module="AdaLN-block",
        module_pattern="transformer_blocks.*.adaln_proj.linear",
        rows=adaln_unique_timesteps,
        in_features=arch.time_embed_dim,
        out_features=6 * arch.hidden_size * arch.modalities,
        dtype=dtype,
        count=arch.num_layers,
        source=src,
        notes="M is number of unique modality timesteps, not token count.",
    )
    add_linear_triplet(
        workloads,
        scope="final",
        module="AdaLN-out",
        module_pattern="norm_out.linear",
        rows=adaln_unique_timesteps,
        in_features=arch.time_embed_dim,
        out_features=2 * arch.hidden_size,
        dtype=dtype,
        count=1,
        source=src,
        notes="Final AdaLN projection; separate from the 50 per-block AdaLN projections.",
    )

    if include_refiner:
        ref_rows = geometry.refiner_local_rows
        for name in ("Q", "K", "V"):
            add_linear_triplet(
                workloads,
                scope="text_refiner",
                module=f"Refiner-{name}",
                module_pattern=f"token_refiner.refiner_blocks.*.attn.to_{name.lower()}",
                rows=ref_rows,
                in_features=arch.hidden_size,
                out_features=inner,
                dtype=dtype,
                count=arch.num_refiner_layers,
                source=src,
                notes="Text-only sequence, separately SP-sharded before refiner blocks.",
            )
        add_linear_triplet(
            workloads,
            scope="text_refiner",
            module="Refiner-O",
            module_pattern="token_refiner.refiner_blocks.*.attn.to_out",
            rows=ref_rows,
            in_features=inner,
            out_features=arch.hidden_size,
            dtype=dtype,
            count=arch.num_refiner_layers,
            source=src,
        )
        add_linear_triplet(
            workloads,
            scope="text_refiner",
            module="Refiner-Gate+Up",
            module_pattern="token_refiner.refiner_blocks.*.ff.fc_in",
            rows=ref_rows,
            in_features=arch.hidden_size,
            out_features=2 * arch.ffn_dim,
            dtype=dtype,
            count=arch.num_refiner_layers,
            source=src,
        )
        add_linear_triplet(
            workloads,
            scope="text_refiner",
            module="Refiner-Down",
            module_pattern="token_refiner.refiner_blocks.*.ff.fc_out",
            rows=ref_rows,
            in_features=arch.ffn_dim,
            out_features=arch.hidden_size,
            dtype=dtype,
            count=arch.num_refiner_layers,
            source=src,
        )

    return workloads


def write_csv(path: Path, workloads: list[Workload]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(workloads[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in workloads:
            writer.writerow(asdict(row))


def write_metadata(
    path: Path,
    *,
    config_path: Path,
    revision: str,
    geometry: Geometry,
    arch: Arch,
    dtype: str,
    checkpointing: str,
    text_source: str,
    adaln_unique_timesteps: int,
    checkpoint_num_layers: int,
    active_num_layers: int,
) -> None:
    metadata = {
        "config": str(config_path),
        "git_revision": revision,
        "dtype": dtype,
        "gradient_checkpointing": checkpointing,
        "text_token_source": text_source,
        "adaln_unique_timesteps": adaln_unique_timesteps,
        "checkpoint_num_layers": checkpoint_num_layers,
        "active_num_layers": active_num_layers,
        "full_model_reference_count": checkpoint_num_layers,
        "architecture": asdict(arch),
        "geometry": asdict(geometry),
        "limitations": [
            "Static collector: runtime strides/contiguity are not observed.",
            "logical_count_per_step does not include activation-checkpoint recompute executions.",
            "Actual BLAS/kernel dispatch may transpose/swap logical GEMM operands internally.",
            "Runtime hooks/Nsight Systems or Nsight Compute should validate the selected shapes.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/train/configs/overfit_minimax_h3_t2va.yaml"),
        help="FastVideo training YAML.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-tokens", type=int, help="Number of valid text tokens in the SFT sample.")
    source.add_argument("--parquet", type=Path, help="Preprocessed T2VA parquet containing text_attention_mask.")
    p.add_argument("--parquet-row", type=int, default=0)
    p.add_argument("--mask-column", default="text_attention_mask")
    p.add_argument("--output", type=Path, default=Path("artifacts/minimax_h3_gemm_workloads.csv"))
    p.add_argument("--metadata-output", type=Path, default=None)
    p.add_argument("--include-refiner", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--adaln-unique-timesteps",
        type=int,
        default=2,
        help="T2VA SFT normally has distinct video/text and audio timesteps, hence 2.",
    )
    p.add_argument("--vae-spatial-downsample", type=int, default=DEFAULT_VAE_SPATIAL_DOWNSAMPLE)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--audio-latents-per-second", type=int, default=DEFAULT_AUDIO_LATENTS_PER_SECOND)
    p.add_argument("--audio-channels", type=int, default=DEFAULT_AUDIO_CHANNELS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    if args.parquet is not None:
        text_tokens, column_used = text_tokens_from_parquet(
            args.parquet, args.parquet_row, args.mask_column)
        text_source = f"{args.parquet}:{column_used}[row={args.parquet_row}]"
    else:
        text_tokens = int(args.text_tokens)
        text_source = "--text-tokens"

    num_latent_t = int(nested_get(config, "training", "data", "num_latent_t"))
    num_frames = int(nested_get(config, "training", "data", "num_frames"))
    height = int(nested_get(config, "training", "data", "num_height"))
    width = int(nested_get(config, "training", "data", "num_width"))
    sp_size = int(nested_get(config, "training", "distributed", "sp_size", default=1))
    dtype = str(nested_get(config, "training", "dit_precision", default="bf16"))
    checkpointing = str(
        nested_get(config, "training", "model", "enable_gradient_checkpointing_type", default="")
        or nested_get(config, "models", "student", "enable_gradient_checkpointing_type", default="")
        or "disabled"
    )

    checkpoint_num_layers = DEFAULT_NUM_LAYERS
    requested_layers = nested_get(config, "models", "student", "num_transformer_layers")
    active_num_layers = checkpoint_num_layers if requested_layers is None else int(requested_layers)
    if not 1 <= active_num_layers <= checkpoint_num_layers:
        raise ValueError(f"num_transformer_layers must satisfy 1 <= N <= {checkpoint_num_layers}, "
                         f"got {requested_layers!r}")
    arch = Arch(num_layers=active_num_layers)
    geometry = build_geometry(
        text_tokens=text_tokens,
        num_latent_t=num_latent_t,
        num_frames=num_frames,
        height=height,
        width=width,
        sp_size=sp_size,
        arch=arch,
        vae_spatial_downsample=args.vae_spatial_downsample,
        fps=args.fps,
        audio_latents_per_second=args.audio_latents_per_second,
        audio_channels=args.audio_channels,
    )
    workloads = build_workloads(
        geometry=geometry,
        arch=arch,
        dtype=dtype,
        include_refiner=args.include_refiner,
        adaln_unique_timesteps=args.adaln_unique_timesteps,
    )

    write_csv(args.output, workloads)
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    revision = resolve_git_revision(args.config.parent)
    write_metadata(
        metadata_path,
        config_path=args.config,
        revision=revision,
        geometry=geometry,
        arch=arch,
        dtype=dtype,
        checkpointing=checkpointing,
        text_source=text_source,
        adaln_unique_timesteps=args.adaln_unique_timesteps,
        checkpoint_num_layers=checkpoint_num_layers,
        active_num_layers=active_num_layers,
    )

    print("=== MiniMax H3 static GEMM workload ===")
    print(f"config:              {args.config}")
    print(f"git revision:        {revision}")
    print(f"text tokens:         {geometry.text_tokens}")
    print(f"video rows:          {geometry.video_rows}")
    print(f"audio rows:          {geometry.audio_rows} ({geometry.audio_latents} latents x {args.audio_channels} ch)")
    print(f"global sequence:     {geometry.global_sequence}")
    print(f"SP padded sequence:  {geometry.padded_global_sequence}")
    print(f"main local M:        {geometry.main_local_rows}")
    print(f"refiner local M:     {geometry.refiner_local_rows}")
    print(f"AdaLN M:             {args.adaln_unique_timesteps}")
    print(f"dtype:               {dtype}")
    print(f"checkpointing:       {checkpointing}")
    print(f"checkpoint layers:   {checkpoint_num_layers}")
    print(f"active layers:       {active_num_layers}")
    print(f"CSV:                 {args.output}")
    print(f"metadata:            {metadata_path}")
    print()
    print("Representative Fwd shapes (logical C[m,n] = A[m,k] @ B[k,n]):")
    seen: set[tuple[str, str]] = set()
    for w in workloads:
        if w.phase != "Fwd" or w.scope != "main_transformer":
            continue
        key = (w.scope, w.module)
        if key in seen:
            continue
        seen.add(key)
        print(
            f"  {w.module:12s} m={w.gemm_m:<6d} n={w.gemm_n:<6d} k={w.gemm_k:<6d} "
            f"count={w.logical_count_per_step}"
        )


if __name__ == "__main__":
    main()
