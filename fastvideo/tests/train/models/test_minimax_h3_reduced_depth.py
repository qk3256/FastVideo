import json
import csv
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.models.loader.component_loader import (
    _minimax_h3_depth_key_filter,
    resolve_minimax_h3_loader_depth,
    resolve_minimax_h3_num_layers,
)
from fastvideo.models.loader.weight_utils import safetensors_weights_iterator


def test_none_keeps_checkpoint_depth():
    assert resolve_minimax_h3_num_layers(None, 50) == 50


def test_hf_merged_config_override_resolves_four_layers():
    # The loader calls this after update_model_arch() has restored 50.
    assert resolve_minimax_h3_num_layers(4, 50) == 4


def test_non_h3_loader_does_not_access_num_layers():
    class Arch:
        pass

    class Config:
        arch_config = Arch()

    assert resolve_minimax_h3_loader_depth("OtherTransformer", 4, Config()) is None


def test_filter_keeps_first_four_refiner_and_outputs():
    keep = _minimax_h3_depth_key_filter(4)
    assert keep("transformer_blocks.0.attn.to_q.weight")
    assert keep("transformer_blocks.3.ff.net.2.weight")
    assert not keep("transformer_blocks.4.attn.to_q.weight")
    assert not keep("transformer_blocks.49.adaln_proj.linear.bias")
    assert keep("token_refiner.refiner_blocks.1.attn.to_q.weight")
    assert keep("proj_out.weight")
    assert keep("audio_proj_out.bias")


def test_filter_runs_before_tensor_read_and_broadcast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "weights.safetensors"
    save_file({
        "transformer_blocks.0.weight": torch.ones(2),
        "transformer_blocks.4.weight": torch.ones(2),
        "proj_out.weight": torch.ones(2),
    }, str(path))
    seen = []
    from safetensors import safe_open as real_safe_open

    class Handle:
        def __enter__(self):
            self.handle = real_safe_open(str(path), framework="pt", device="cpu")
            return self

        def __exit__(self, *args):
            self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_tensor(self, name):
            seen.append(name)
            return self.handle.get_tensor(name)

    monkeypatch.setattr("fastvideo.models.loader.weight_utils.safe_open", lambda *a, **k: Handle())
    values = list(safetensors_weights_iterator([str(path)], to_cpu=True, key_filter=_minimax_h3_depth_key_filter(4)))
    assert {name for name, _ in values} == {"proj_out.weight", "transformer_blocks.0.weight"}
    assert "transformer_blocks.4.weight" not in seen


def test_invalid_depths_fail_clearly():
    for value in (0, -1, 51, True, "4"):
        with pytest.raises(ValueError, match="num_transformer_layers"):
            resolve_minimax_h3_num_layers(value, 50)


def test_strict_load_rejects_retained_missing_key():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        module.load_state_dict({"weight": torch.ones_like(module.weight)}, strict=True)


def test_shape_collector_metadata_contract(tmp_path: Path):
    config = tmp_path / "profile.yaml"
    config.write_text("""
models:
  student:
    num_transformer_layers: 4
training:
  data: {num_latent_t: 1, num_frames: 1, num_height: 32, num_width: 32}
  distributed: {sp_size: 2}
  dit_precision: bf16
""")
    # Exercise the collector's config contract without requiring parquet data.
    import subprocess
    out = tmp_path / "out.csv"
    meta = tmp_path / "out.json"
    subprocess.run([
        sys.executable, "tools/minimax_h3_shape_collector.py", "--config", str(config),
        "--text-tokens", "2", "--output", str(out), "--metadata-output", str(meta),
    ], check=True)
    payload = json.loads(meta.read_text())
    assert payload["active_num_layers"] == 4
    assert payload["checkpoint_num_layers"] == 50
    assert payload["full_model_reference_count"] == 50
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    main = next(row for row in rows if row["scope"] == "main_transformer" and row["module"] == "Q")
    refiner = next(row for row in rows if row["scope"] == "text_refiner" and row["module"] == "Refiner-Q")
    final = next(row for row in rows if row["scope"] == "final" and row["module"] == "AdaLN-out")
    assert main["logical_count_per_step"] == "4"
    assert main["full_model_reference_count"] == "50"
    assert refiner["logical_count_per_step"] == "2"
    assert refiner["full_model_reference_count"] == "2"
    assert final["logical_count_per_step"] == "1"
    assert final["full_model_reference_count"] == "1"


def _wrapper_training_config():
    """Minimal hermetic stand-in for the TrainingConfig fields the wrapper reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        data=SimpleNamespace(
            train_batch_size=1,
            training_cfg_rate=0.0,
            preprocessed_data_type="t2va",
        ),
        model=SimpleNamespace(enable_gradient_checkpointing_type=None),
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(uniform_parameter_dtype=False),
            text_encoder_configs=[SimpleNamespace()],
        ),
    )


def test_constructor_forwards_num_transformer_layers_to_loader(monkeypatch: pytest.MonkeyPatch):
    """The YAML/constructor depth must reach ``load_module_from_path`` unchanged."""
    from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model

    captured: dict = {}

    def fake_load_module_from_path(**kwargs):
        captured.update(kwargs)
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(
        "fastvideo.train.models.minimax_h3.minimax_h3.load_module_from_path",
        fake_load_module_from_path,
    )
    model = MiniMaxH3Model(
        init_from="unused",
        training_config=_wrapper_training_config(),
        trainable=False,
        num_transformer_layers=4,
    )
    assert captured["num_transformer_layers"] == 4
    assert captured["module_type"] == "transformer"
    assert isinstance(model.transformer, torch.nn.Linear)

    # Unset depth must forward as None so the loader keeps full checkpoint depth.
    MiniMaxH3Model(
        init_from="unused",
        training_config=_wrapper_training_config(),
        trainable=False,
    )
    assert captured["num_transformer_layers"] is None
