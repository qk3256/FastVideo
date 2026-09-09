# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the runtime shape collector callback."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fastvideo.train.callbacks.runtime_shape_collector import RuntimeShapeCollectorCallback


class _TinyFF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_in = nn.Linear(8, 32, bias=False)
        self.fc_out = nn.Linear(32, 8, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = _TinyFF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff.fc_out(torch.nn.functional.silu(self.ff.fc_in(x)))


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_TinyBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.transformer_blocks[0], x, use_reentrant=False)


def _fake_method(transformer: nn.Module) -> SimpleNamespace:
    method = SimpleNamespace(student=SimpleNamespace(transformer=transformer))
    method.backward = lambda loss, *_a, **_k: loss.backward()
    return method


def test_collector_observes_fwd_recompute_dgrad_wgrad(tmp_path: Path):
    transformer = _TinyTransformer()
    method = _fake_method(transformer)
    cb = RuntimeShapeCollectorCallback(output_dir=str(tmp_path), enabled=True)
    cb.on_train_start(method, iteration=0)

    for step in (1, 2):
        x = torch.randn(4, 8, requires_grad=True)
        loss = transformer(x).sum()
        method.backward(loss)
        cb.on_training_step_end(method, {"total_loss": float(loss.detach())}, iteration=step)

    cb.on_train_end(method, iteration=2)

    lines = (tmp_path / "hooks.rank0.jsonl").read_text().strip().splitlines()
    assert lines, "no hook records flushed"
    records = [json.loads(line) for line in lines]
    path = "transformer_blocks.0.ff.fc_in"
    phases = [(r["phase"], r["step"]) for r in records if r["module_path"] == path]
    assert ("Fwd", 1) in phases and ("RecomputeFwd", 1) in phases
    assert ("Dgrad", 1) in phases and ("Wgrad", 1) in phases
    fwd = next(r for r in records if r["module_path"] == path and r["phase"] == "Fwd")
    assert fwd["input"]["shape"] == [4, 8]
    assert fwd["weight"]["shape"] == [32, 8]
    assert fwd["input"]["align16"] is True
    wgrad = next(r for r in records if r["module_path"] == path and r["phase"] == "Wgrad" and r["step"] == 1)
    assert wgrad["wgrad_operands_paired"] is True
    assert wgrad["wgrad_lhs_meta"]["shape"] == [4, 8]
    dgrad = next(r for r in records if r["module_path"] == path and r["phase"] == "Dgrad" and r["step"] == 1)
    assert dgrad["input"]["shape"] == [4, 32]  # dY
    assert dgrad["output"]["shape"] == [4, 8]  # dX

    report = json.loads((tmp_path / "collector_report.rank0.json").read_text())
    assert report["all_targets_complete"] is True
    assert report["recompute_observed_anywhere"] is True
    assert report["wgrad_paired_everywhere"] is True
    assert report["module_checks"][path]["observed_call_count"]["Fwd"] == 2  # initial forwards only
    assert report["module_checks"][path]["observed_call_count"]["RecomputeFwd"] == 2
    # Segment-tail module: autograd does not need its forward replayed, but its
    # Wgrad must still pair with the initial-forward input metadata.
    tail = report["module_checks"]["transformer_blocks.0.ff.fc_out"]
    assert tail["observed_call_count"]["Fwd"] >= 2
    assert tail["observed_call_count"]["Wgrad"] == 2
    assert tail["wgrad_operands_paired"] is True


def test_collector_disabled_is_noop(tmp_path: Path):
    method = _fake_method(_TinyTransformer())
    cb = RuntimeShapeCollectorCallback(output_dir=str(tmp_path), enabled=False)
    cb.on_train_start(method, iteration=0)
    assert not (tmp_path / "hooks.rank0.jsonl").exists()
    # disabled callback must not wrap method.backward
    assert "flag" not in getattr(method.backward, "__name__", "")


class _TupleLinear(nn.Module):
    """Mimics FastVideo linear convention: forward returns (tensor, bias)."""

    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        with torch.no_grad():
            self.weight = torch.nn.Parameter(torch.randn(out_f, in_f) * 0.02)

    def forward(self, x: torch.Tensor):
        return x @ self.weight.t(), None


class _TupleFF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_in = _TupleLinear(8, 32)
        self.fc_out = _TupleLinear(32, 8)


class _TupleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = _TupleFF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.ff.fc_in(x)
        out, _ = self.ff.fc_out(torch.nn.functional.silu(y))
        return out


class _TupleTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_TupleBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.transformer_blocks[0], x, use_reentrant=False)


def test_collector_unpacks_tuple_output_and_records_real_wgrad_grad(tmp_path: Path):
    transformer = _TupleTransformer()
    method = _fake_method(transformer)
    cb = RuntimeShapeCollectorCallback(output_dir=str(tmp_path), enabled=True)
    cb.on_train_start(method, iteration=0)

    x = torch.randn(4, 8, requires_grad=True)
    loss = transformer(x).sum()
    method.backward(loss)
    cb.on_before_optimizer_step(method, iteration=1)
    cb.on_training_step_end(method, {"total_loss": float(loss.detach())}, iteration=1)
    cb.on_train_end(method, iteration=1)

    records = [json.loads(l) for l in (tmp_path / "hooks.rank0.jsonl").read_text().splitlines()]
    fwd = next(r for r in records if r["module_path"].endswith("ff.fc_in") and r["phase"] == "Fwd")
    assert fwd["output"] is not None
    assert fwd["output"]["shape"] == [4, 32]
    assert fwd["output"]["stride"] is not None and fwd["output"]["align256"] is not None

    wgrad = next(r for r in records if r["module_path"].endswith("ff.fc_in") and r["phase"] == "Wgrad")
    assert wgrad.get("wgrad_output_is_real_grad") is True
    assert wgrad["output"]["shape"] == [32, 8]
    assert wgrad["output"]["stride"] is not None
    assert wgrad["output"]["align16"] is not None
