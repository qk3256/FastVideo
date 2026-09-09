#!/usr/bin/env python3
"""Raw GEMM replay from the MiniMax H3 runtime shape corpus (P5).

Standalone single-GPU process; no FSDP, no dataloader, no model state.
Rebuilds every observed layout per (scope, op_role) from the runtime hook
JSONLs — Fwd X@W^T, Dgrad dY@W, Wgrad dY^T@X — using runtime-observed dtypes.
The training wrapper runs the transformer forward under bf16 autocast, so an
fp32-recorded forward input executes as a bf16 GEMM (executed_dtype says so).

Phase A: every deduped case, warmup 10 / repeat 30, plus kernel-name capture.
Phase B: top-5 by FLOPs, warmup 20 / repeat 100, plus kernel-name capture.
Every row carries workload_id, transpose layout, operand strides/alignment.
Wgrad replays X with the runtime-paired layout from wgrad_lhs_meta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

_GEMM_HINTS = ("gemm", "nvjet", "cutlass", "s16816", "s1688", "h884", "16816")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def workload_id(scope: str, role: str, kind: str, m: int, n: int, k: int, dtype: str) -> str:
    return f"{scope}/{role}/{kind}/{m}x{n}x{k}/{dtype}"


def replicate_layout(shape: list[int], stride: list[int], dtype: torch.dtype, device) -> torch.Tensor:
    """Allocate a tensor whose shape and stride match a runtime observation."""
    numel = 1
    for s, st in zip(shape, stride):
        if s > 0:
            numel = max(numel, (s - 1) * abs(st) + 1)
    base = torch.empty(numel, device=device, dtype=dtype).normal_()
    return base.as_strided(shape, stride)


def load_cases(hooks_dir: Path) -> list[dict]:
    """Rebuild deduped GEMM cases from runtime Fwd/Dgrad/Wgrad records."""
    records: dict[tuple[str, str], dict] = {}
    wgrad_lhs: dict[str, dict] = {}
    for rank in (0, 1):
        path = hooks_dir / f"hooks.rank{rank}.jsonl"
        for line in path.read_text().splitlines():
            e = json.loads(line)
            records[(e["module_path"], e["phase"])] = e  # same across steps/ranks; keep last id
            if e["phase"] == "Wgrad" and e.get("wgrad_lhs_meta"):
                wgrad_lhs[e["module_path"]] = e["wgrad_lhs_meta"]

    cases: dict[tuple, dict] = {}

    def add_case(role: str, scope: str, kind: str, m: int, n: int, k: int,
                 dtype: str, src_module: str, phase: str) -> dict:
        key = (scope, role, kind, m, n, k, dtype)
        if key not in cases:
            cases[key] = {
                "scope": scope, "op_role": role, "kind": kind,
                "m": m, "n": n, "k": k, "dtype": dtype,
                "src_module": src_module, "src_phases": sorted({phase}),
                "workload_id": workload_id(scope, role, kind, m, n, k, dtype),
            }
        else:
            cases[key]["src_phases"] = sorted(set(cases[key]["src_phases"]) | {phase})
        return cases[key]

    for (module_path, phase), e in records.items():
        role, scope = e["op_role"], e["scope"]
        if phase == "Fwd" and e["input"] and e["weight"]:
            k = e["input"]["shape"][-1]
            m = math.prod(e["input"]["shape"][:-1])
            n, k_w = e["weight"]["shape"]
            assert k == k_w, (module_path, k, k_w)
            c = add_case(role, scope, "Fwd", m, n, k, e["input"]["dtype"], module_path, phase)
            c["layout"] = "NT"  # X[M,K] @ W^T  (stored W = [N,K])
        elif phase == "Dgrad" and e["input"] and e["weight"]:
            grad_dtype = e["input"]["dtype"]
            n_w, k_w = e["weight"]["shape"]
            dy_n = e["input"]["shape"][-1]
            m_tokens = math.prod(e["input"]["shape"][:-1])
            assert dy_n == n_w, (module_path, dy_n, n_w)
            c = add_case(role, scope, "Dgrad", m_tokens, k_w, n_w, grad_dtype, module_path, phase)
            c["layout"] = "NN"  # dY[M,N] @ W[N,K]
        elif phase == "Wgrad" and e["input"] and e["weight"]:
            grad_dtype = e["input"]["dtype"]
            n_w, k_w = e["weight"]["shape"]
            dy_n = e["input"]["shape"][-1]
            assert dy_n == n_w, (module_path, dy_n, n_w)
            m_tokens = math.prod(e["input"]["shape"][:-1])
            c = add_case(role, scope, "Wgrad", n_w, k_w, m_tokens, grad_dtype, module_path, phase)
            c["layout"] = "TN"  # dY^T[N,M] @ X[M,K]
            lhs = wgrad_lhs.get(module_path)
            if lhs and lhs.get("shape") and lhs.get("stride"):
                shape = list(lhs["shape"])
                x_flat_shape = [math.prod(shape[:-1]), shape[-1]]
                # runtime stride over flattened token axis
                if len(shape) == 3 and list(lhs["stride"][-3:])[:2] == [lhs["stride"][-2], lhs["stride"][-1]]:
                    flat_stride = [lhs["stride"][-2], lhs["stride"][-1]]
                else:
                    flat_stride = [x_flat_shape[-1], 1]
                c["x_layout"] = {"shape": x_flat_shape, "stride": flat_stride,
                                 "contiguous": lhs.get("contiguous", True)}
    return list(cases.values())


def _flops(case: dict) -> int:
    return 2 * case["m"] * case["n"] * case["k"]


def make_operands(case: dict, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor, callable]:
    m, n, k = case["m"], case["n"], case["k"]
    if case["kind"] == "Fwd":
        x = torch.randn(m, k, device=device, dtype=dtype)
        w = torch.randn(n, k, device=device, dtype=dtype)
        run = lambda: x @ w.t()  # noqa: E731
    elif case["kind"] == "Dgrad":
        x = torch.randn(m, k, device=device, dtype=dtype)  # dY[M,N] with N under case k slot
        w = torch.randn(k, n, device=device, dtype=dtype)
        run = lambda: x @ w  # noqa: E731
    else:  # Wgrad: dY^T[N,M] @ X[M,K]
        dy = torch.randn(k, m, device=device, dtype=dtype)
        x_layout = case.get("x_layout")
        if x_layout and not x_layout.get("contiguous", True):
            x = replicate_layout(x_layout["shape"], x_layout["stride"], dtype, device)
        else:
            x = torch.randn(k, n, device=device, dtype=dtype)  # X[M,K] with M under case k slot
        run = lambda: dy.t() @ x  # noqa: E731
        return dy, x, run
    return x, w, run


def operand_meta(t: torch.Tensor) -> dict:
    ptr = t.data_ptr()
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "contiguous": bool(t.is_contiguous()),
        "align16": ptr % 16 == 0,
        "align256": ptr % 256 == 0,
    }


def bench_case(case: dict, warmup: int, repeat: int, device: torch.device) -> dict:
    dtype = torch.bfloat16 if case["dtype"] in ("bfloat16", "bf16") else torch.float32
    a, b, run = make_operands(case, dtype, device)

    def run_once() -> None:
        # Faithful to training: the forward pass runs under bf16 autocast.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            run()

    torch.cuda.synchronize(device)
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    times.sort()
    med = times[len(times) // 2]
    p20 = times[max(0, int(len(times) * 0.2) - 1)]
    p80 = times[min(len(times) - 1, int(len(times) * 0.8))]
    tflops = _flops(case) / (med * 1e-3) / 1e12
    return {
        "median_ms": med, "p20_ms": p20, "p80_ms": p80, "tflops": tflops,
        "warmup": warmup, "repeat": repeat,
        "executed_dtype": "bfloat16(autocast)" if case["dtype"] == "float32" else case["dtype"],
        "lhs": operand_meta(a), "rhs": operand_meta(b),
    }


def capture_kernel(case: dict, device: torch.device) -> tuple[str, str]:
    dtype = torch.bfloat16 if case["dtype"] in ("bfloat16", "bf16") else torch.float32
    _, _, run = make_operands(case, dtype, device)

    def run_once() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            run()

    for _ in range(3):
        run_once()
    torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run_once()
        torch.cuda.synchronize(device)
    kernels = sorted(
        (e for e in prof.key_averages()
         if e.device_type == torch.autograd.DeviceType.CUDA
         and any(h in e.key.lower() for h in _GEMM_HINTS)),
        key=lambda e: -getattr(e, "self_device_time_total", 0.0))
    name = kernels[0].key if kernels else ""
    lower = name.lower()
    backend = ("cublaslt" if "nvjet" in lower else "cutlass/cublaslt" if "cutlass" in lower
               else "cublas" if name else "unknown")
    return name, backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooks-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    cases = load_cases(args.hooks_dir)
    print(f"deduped cases: {len(cases)}")

    rows_a = []
    for case in cases:
        stats = bench_case(case, warmup=10, repeat=30, device=device)
        kernel, backend = capture_kernel(case, device)
        rows_a.append({**case, **stats, "phase": "A", "flops": _flops(case),
                       "kernel_name": kernel, "kernel_backend": backend})
        print("A %s median=%.3fms %.1fTF/s kernel=%s" % (
            case["workload_id"], stats["median_ms"], stats["tflops"], kernel or "n/a"))
    _dump(args.output_dir / "raw_gemm_phaseA.jsonl", rows_a)

    order = sorted(rows_a, key=lambda r: -r["flops"])
    rows_b = []
    for case in order[: args.top]:
        stats = bench_case(case, warmup=20, repeat=100, device=device)
        kernel, backend = capture_kernel(case, device)
        rows_b.append({**(case | stats), "phase": "B", "flops": _flops(case),
                       "kernel_name": kernel, "kernel_backend": backend})
        print("B %s median=%.3fms %.1fTF/s" % (case["workload_id"], stats["median_ms"], stats["tflops"]))
    _dump(args.output_dir / "raw_gemm_phaseB.jsonl", rows_b)
    meta = {
        "source_hooks_dir": str(args.hooks_dir),
        "source_hooks_sha256": {f"hooks.rank{r}.jsonl": sha256_file(args.hooks_dir / f"hooks.rank{r}.jsonl")
                                for r in (0, 1)},
        "cases": len(cases),
        "kernel_captured": sum(1 for r in rows_a + rows_b if r["kernel_name"]),
        "device": torch.cuda.get_device_name(device),
        "note": "raw GEMM replay isolated from full-step training: no FSDP, no dataloader, "
                "no model state; bf16-autocast faithful to the training wrapper",
    }
    (args.output_dir / "raw_gemm_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("raw GEMM replay done")


def _dump(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
