# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the one-step torch profiler callback."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from fastvideo.train.callbacks.torch_profiler import TorchProfilerCallback


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.ModuleDict({"ff": nn.ModuleDict({
            "fc_in": nn.Linear(8, 32, bias=False),
            "fc_out": nn.Linear(32, 8, bias=False),
        })})])
        self.transformer = self  # callback hooks named_modules on .transformer

    def prepare_batch(self, x):
        return x * 2

    def predict_noise(self, x):
        blk = self.transformer_blocks[0]["ff"]
        return blk["fc_out"](torch.nn.functional.silu(blk["fc_in"](x)))


def _fake_method() -> SimpleNamespace:
    student = _TinyStudent()
    method = SimpleNamespace(student=student)
    def backward(loss):
        loss.backward()
    method.backward = backward
    method.optimizers_schedulers_step = lambda step: None
    return method


def test_profiler_callback_emits_trace_and_exports(tmp_path: Path):
    method = _fake_method()
    cb = TorchProfilerCallback(output_dir=str(tmp_path), enabled=True)
    cb.on_train_start(method, iteration=0)
    for step in (1, 2, 3):
        x = torch.randn(4, 8, requires_grad=True)
        batch = method.student.prepare_batch(x)
        out = method.student.predict_noise(batch)
        loss = out.sum()
        method.backward(loss)
        method.optimizers_schedulers_step(step)
        cb.on_training_step_end(method, {"total_loss": float(loss.detach())}, iteration=step)
    cb.on_train_end(method, iteration=3)

    trace = tmp_path / "trace.rank0.json"
    assert trace.exists() and trace.stat().st_size > 0
    meta = json.loads((tmp_path / "profiler_meta.rank0.json").read_text())
    assert meta["profiled_step"] == 3
    assert meta["schedule"] == {"wait": 1, "warmup": 1, "active": 1}
    assert meta["profile_memory"] is True and meta["record_shapes"] and meta["with_flops"]
    assert not meta["with_stack"]
    for name in ("cuda_top200.csv", "cpu_top200.csv", "cuda_self_time.txt", "cpu_self_time.txt"):
        assert (tmp_path / f"trace.rank0.{name}").exists()
    text = trace.read_text()
    assert "prepare_batch" in text and "forward" in text and "backward" in text
    assert "main_transformer/FFN-GateUp" in text  # per-module NVTX range present


def test_profiler_disabled_is_noop(tmp_path: Path):
    method = _fake_method()
    cb = TorchProfilerCallback(output_dir=str(tmp_path), enabled=False)
    cb.on_train_start(method, iteration=0)
    cb.on_train_end(method, iteration=0)
    assert not list(tmp_path.glob("trace.*"))
