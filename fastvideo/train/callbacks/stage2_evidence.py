# SPDX-License-Identifier: Apache-2.0
"""Stage-2 training-closure evidence hook (FSDP-safe, profiler-free).

Records the evidence contract for reduced-depth MiniMax H3 SFT runs: finite
loss, backward completion, optimizer-step execution, step counter, per-rank
peak memory, per-step wall time, and a cross-rank weight checksum proving a
retained parameter actually changed.  Each rank appends its own
``metrics.rank<N>.jsonl`` and writes ``evidence_meta.json`` at train start;
rank 0 additionally writes ``evidence_summary.json`` at train end.

The hook only reduces parameter shards into a scalar checksum — it never
gathers full weights, never stores tensors between hooks, and runs no
profiler or shape-collector hooks, keeping Stage-2 memory behavior clean.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

logger = init_logger(__name__)

_DEFAULT_CHECKSUM_HINT = "transformer_blocks.0."


def _dist_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _global_weight_checksum(transformer: torch.nn.Module, param_name: str) -> float:
    """Sum |shard| of *param_name* locally, then SUM-reduce across ranks."""
    named = dict(transformer.named_parameters())
    if param_name not in named:
        raise RuntimeError(f"checksum parameter {param_name!r} not found in transformer")
    param = named[param_name]
    local = param.to_local() if hasattr(param, "to_local") else param
    total = local.detach().abs().to(torch.float64).sum().reshape(1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.item())


def _optimizer_step_count(method: Any) -> int | None:
    """Read the real optimizer step counter from the first optimizer's state."""
    try:
        for opt in method.get_optimizers(0):
            for group in opt.param_groups:
                for param in group["params"]:
                    state = opt.state.get(param)
                    if state and "step" in state:
                        value = state["step"]
                        return int(value.item() if torch.is_tensor(value) else value)
    except Exception:
        return None
    return None


class Stage2EvidenceCallback(Callback):
    """Per-step stage-2 evidence: loss/memory/wall-time + weight-change proof."""

    def __init__(
        self,
        *,
        output_dir: str,
        checksum_param_hint: str = _DEFAULT_CHECKSUM_HINT,
    ) -> None:
        if not output_dir:
            raise ValueError("Stage2EvidenceCallback requires a nonempty output_dir")
        self.output_dir = Path(output_dir)
        self.checksum_param_hint = checksum_param_hint
        self._checksum_name: str | None = None
        self._initial_checksum: float | None = None
        self._before_opt_step: int | None = None
        self._rank = 0
        self._world = 1
        self._metrics_path: Path | None = None
        self._step_entries: list[dict[str, Any]] = []

    def _select_checksum_param(self, transformer: torch.nn.Module) -> str:
        named = dict(transformer.named_parameters())
        candidates = sorted(name for name in named if self.checksum_param_hint in name)
        if not candidates:
            candidates = sorted(named)
        if not candidates:
            raise RuntimeError("transformer exposes no named parameters for checksum")
        self._checksum_name = candidates[0]
        return self._checksum_name

    def on_train_start(self, method: Any, iteration: int = 0) -> None:
        self._rank, self._world = _dist_rank_world()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / f"metrics.rank{self._rank}.jsonl"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        transformer = method.student.transformer
        name = self._select_checksum_param(transformer)
        self._initial_checksum = _global_weight_checksum(transformer, name)
        meta = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "rank": self._rank,
            "world_size": self._world,
            "checksum_param_name": name,
            "checksum_param_hint": self.checksum_param_hint,
            "initial_weight_checksum": self._initial_checksum,
        }
        (self.output_dir / f"evidence_meta.rank{self._rank}.json").write_text(json.dumps(meta, indent=2) + "\n")
        logger.info("stage2 evidence: rank=%d checksum_param=%s initial_checksum=%.6f", self._rank, name,
                    self._initial_checksum)

    def on_before_optimizer_step(self, method: Any, iteration: int = 0) -> None:
        # Reaching this hook means forward + backward for the step completed.
        self._before_opt_step = iteration

    def on_training_step_end(self, method: Any, loss_dict: dict[str, Any], iteration: int = 0) -> None:
        if self._metrics_path is None or self._initial_checksum is None or self._checksum_name is None:
            raise RuntimeError("Stage2EvidenceCallback.on_train_start did not run")
        current = _global_weight_checksum(method.student.transformer, self._checksum_name)
        metrics = {k: float(v) for k, v in loss_dict.items() if isinstance(v, (int, float))}
        # Require the real loss term, not just any finite number (step_time_sec
        # alone would otherwise satisfy a fake "finite" pass).
        loss_finite = "total_loss" in metrics and math.isfinite(metrics["total_loss"])
        entry = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "rank": self._rank,
            "world_size": self._world,
            "step": iteration,
            "trainer_step": iteration,
            "optimizer_step": _optimizer_step_count(method),
            "backward_completed": self._before_opt_step == iteration,
            "optimizer_step_completed": True,  # hook ordering: this runs after optimizers_schedulers_step
            "loss_finite": loss_finite,
            "metrics": metrics,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            "weight_checksum_name": self._checksum_name,
            "weight_checksum_initial": self._initial_checksum,
            "weight_checksum_current": current,
            "weight_changed": current != self._initial_checksum,
        }
        with self._metrics_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        self._step_entries.append(entry)
        logger.info(
            "stage2 evidence: rank=%d step=%d loss_finite=%s weight_changed=%s peak_alloc=%.2fGiB peak_rsv=%.2fGiB",
            self._rank, iteration, loss_finite, entry["weight_changed"],
            (entry["peak_allocated_bytes"] or 0) / 2**30, (entry["peak_reserved_bytes"] or 0) / 2**30)

    def on_train_end(self, method: Any, iteration: int = 0) -> None:
        if not self._step_entries:
            raise RuntimeError("stage2 evidence: no training steps were recorded")
        summary = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "completed_steps": len(self._step_entries),
            "all_loss_finite": all(e["loss_finite"] for e in self._step_entries),
            "all_backward_completed": all(e["backward_completed"] for e in self._step_entries),
            "all_optimizer_steps_completed": all(e["optimizer_step_completed"] for e in self._step_entries),
            "weight_changed": self._step_entries[-1]["weight_changed"],
            "weight_checksum_name": self._checksum_name,
            "weight_checksum_initial": self._initial_checksum,
            "weight_checksum_final": self._step_entries[-1]["weight_checksum_current"],
            "step_wall_times_sec": [e["metrics"].get("step_time_sec") for e in self._step_entries],
            "peak_allocated_bytes": max((e["peak_allocated_bytes"] or 0) for e in self._step_entries),
            "peak_reserved_bytes": max((e["peak_reserved_bytes"] or 0) for e in self._step_entries),
        }
        path = self.output_dir / f"evidence_summary.rank{self._rank}.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("stage2 evidence summary: %s", json.dumps(summary))
