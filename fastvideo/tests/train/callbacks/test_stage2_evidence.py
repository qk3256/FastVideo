# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the stage-2 evidence callback."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.callbacks.stage2_evidence import Stage2EvidenceCallback


def _fake_method() -> SimpleNamespace:
    transformer = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    return SimpleNamespace(student=SimpleNamespace(transformer=transformer))


def _run_step(cb: Stage2EvidenceCallback, method: SimpleNamespace, step: int) -> None:
    cb.on_before_optimizer_step(method, iteration=step)
    cb.on_training_step_end(method, {"total_loss": 1.0 / step, "step_time_sec": 0.25}, iteration=step)


def test_evidence_records_weight_change_and_step_fields(tmp_path: Path):
    method = _fake_method()
    cb = Stage2EvidenceCallback(output_dir=str(tmp_path))
    cb.on_train_start(method, iteration=0)

    meta = json.loads((tmp_path / "evidence_meta.rank0.json").read_text())
    assert meta["checksum_param_name"].startswith("0.") or "0." in meta["checksum_param_name"]
    assert isinstance(meta["initial_weight_checksum"], float)

    param = dict(method.student.transformer.named_parameters())[meta["checksum_param_name"]]
    with torch.no_grad():
        param.add_(1.0)
    _run_step(cb, method, 1)
    _run_step(cb, method, 2)

    lines = (tmp_path / "metrics.rank0.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 1
    assert first["optimizer_step_counter"] == 1
    assert first["backward_completed"] is True
    assert first["optimizer_step_completed"] is True
    assert first["loss_finite"] is True
    assert first["weight_changed"] is True
    assert first["weight_checksum_current"] != pytest.approx(first["weight_checksum_initial"])

    cb.on_train_end(method, iteration=2)
    summary = json.loads((tmp_path / "evidence_summary.rank0.json").read_text())
    assert summary["completed_steps"] == 2
    assert summary["all_loss_finite"] is True
    assert summary["all_backward_completed"] is True
    assert summary["weight_changed"] is True


def test_evidence_flags_unchanged_weight_and_nonfinite_loss(tmp_path: Path):
    method = _fake_method()
    cb = Stage2EvidenceCallback(output_dir=str(tmp_path))
    cb.on_train_start(method, iteration=0)
    cb.on_training_step_end(method, {"total_loss": float("nan")}, iteration=1)

    entry = json.loads((tmp_path / "metrics.rank0.jsonl").read_text().strip())
    assert entry["weight_changed"] is False
    assert entry["loss_finite"] is False
    assert entry["backward_completed"] is False


def test_checksum_param_hint_prefers_retained_block_zero(tmp_path: Path):
    method = _fake_method()
    cb = Stage2EvidenceCallback(output_dir=str(tmp_path), checksum_param_hint="1.weight")
    cb.on_train_start(method, iteration=0)
    meta = json.loads((tmp_path / "evidence_meta.rank0.json").read_text())
    assert meta["checksum_param_name"] == "1.weight"


def test_train_end_without_steps_fails(tmp_path: Path):
    method = _fake_method()
    cb = Stage2EvidenceCallback(output_dir=str(tmp_path))
    cb.on_train_start(method, iteration=0)
    with pytest.raises(RuntimeError, match="no training steps"):
        cb.on_train_end(method, iteration=0)
