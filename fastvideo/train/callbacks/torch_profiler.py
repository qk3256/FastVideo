# SPDX-License-Identifier: Apache-2.0
"""PyTorch profiler callback for the H3 profiling queue (P3).

Profiles exactly one stable step of the main reduced-depth configuration
(schedule wait=1, warmup=1, active=1 -> step 3 is the traced step), with
CPU+CUDA activities, record_shapes, with_flops and profile_memory. NVTX
record_function ranges wrap prepare_batch / forward / backward / optimizer,
plus per-target-module forward ranges for Q/K/V/O, FFN gate-up/down and the
AdaLN projections. Per-rank Chrome traces and key_averages exports (CUDA self
time and CPU self time, sorted) land in output_dir. with_stack stays off to
keep traces small. Disabled by default; enable and point output_dir via CLI
overrides. If profile_memory OOMs, retry with
--callbacks.torch_profiler.profile_memory false.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.callbacks.runtime_shape_collector import _role_of, _scope_of

logger = init_logger(__name__)


def _export_key_averages(prof: Any, path_prefix: Path) -> dict[str, int]:
    """Dump key_averages sorted by CUDA/CPU self time as JSON + CSV."""
    base = str(path_prefix)
    Path(base + ".cuda_self_time.txt").write_text(
        prof.key_averages().table(sort_by="self_device_time_total", row_limit=-1) + "\n")
    Path(base + ".cpu_self_time.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1) + "\n")
    counts: dict[str, int] = {}
    for sort_by, name in (("self_device_time_total", "cuda"), ("self_cpu_time_total", "cpu")):
        events = sorted(prof.key_averages(), key=lambda e: -getattr(e, sort_by, 0.0))
        out_rows = []
        for e in events[:200]:
            out_rows.append({
                "key": e.key, "count": e.count,
                "self_device_time_us": getattr(e, "self_device_time_total", 0.0),
                "self_cpu_time_us": getattr(e, "self_cpu_time_total", 0.0),
                "flops": getattr(e, "flops", 0),
            })
        with open(base + f".{name}_top200.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        counts[name] = len(out_rows)
    return counts


class TorchProfilerCallback(Callback):
    """One-step CPU+CUDA profiler with NVTX phase ranges for H3 runs."""

    def __init__(self, *, output_dir: str, enabled: bool = False, profile_memory: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self.profile_memory = bool(profile_memory)
        self._rank = 0
        self._prof: Any = None
        self._records = 0
        self._unwind: list[Any] = []
        self._hooks: list[Any] = []
        self._ranges: list[Any] = []
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _on_trace_ready(self, prof: Any) -> None:
        trace_path = self.output_dir / f"trace.rank{self._rank}"
        prof.export_chrome_trace(str(trace_path) + ".json")
        counts = _export_key_averages(prof, trace_path)
        size = (self.output_dir / f"trace.rank{self._rank}.json").stat().st_size
        meta = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "rank": self._rank,
            "trace_file": f"trace.rank{self._rank}.json",
            "trace_size_bytes": size,
            "schedule": {"wait": 1, "warmup": 1, "active": 1},
            "profiled_step": 3,
            "profile_memory": self.profile_memory,
            "record_shapes": True,
            "with_flops": True,
            "with_stack": False,
            "key_average_rows": counts,
        }
        (self.output_dir / f"profiler_meta.rank{self._rank}.json").write_text(json.dumps(meta, indent=2) + "\n")
        logger.info("profiler trace written: %s (%d bytes)", trace_path, size)

    def _wrap_record_function(self, obj: Any, attr: str, label: str) -> None:
        original = getattr(obj, attr)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)

        setattr(obj, attr, wrapped)
        self._unwind.append((obj, attr, original))

    def on_train_start(self, method: Any, iteration: int = 0) -> None:
        if not self.enabled:
            return
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
        if self._rank == 0:
            (self.output_dir / "stdout.marker").write_text("profiler enabled\n")
        student = method.student
        self._wrap_record_function(student, "prepare_batch", "prepare_batch")
        self._wrap_record_function(student, "predict_noise", "forward")
        self._wrap_record_function(method, "backward", "backward")
        self._wrap_record_function(method, "optimizers_schedulers_step", "optimizer")
        transformer = student.transformer
        for path, module in transformer.named_modules():
            role = _role_of(path)
            if role is None:
                continue
            label = f"{_scope_of(path)}/{role}/{path}"

            def pre_hook(mod: torch.nn.Module, inputs: tuple, _label: str = label) -> None:
                self._ranges.append(torch.profiler.record_function(_label))
                self._ranges[-1].__enter__()

            def post_hook(mod: torch.nn.Module, inputs: tuple, output: Any, _label: str = label) -> None:
                if self._ranges and self._ranges[-1] is not None:
                    rng = self._ranges.pop()
                    try:
                        rng.__exit__(None, None, None)
                    except Exception:
                        pass

            self._hooks.append(module.register_forward_pre_hook(pre_hook))
            self._hooks.append(module.register_forward_hook(post_hook))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=self._on_trace_ready,
            record_shapes=True,
            with_flops=True,
            profile_memory=self.profile_memory,
            with_stack=False,
        )
        self._prof.__enter__()
        logger.info("torch profiler armed (wait=1 warmup=1 active=1) on rank %d", self._rank)

    def on_training_step_end(self, method: Any, loss_dict: dict[str, Any], iteration: int = 0) -> None:
        if self._prof is not None:
            self._prof.step()

    def on_train_end(self, method: Any, iteration: int = 0) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for obj, attr, original in self._unwind:
            setattr(obj, attr, original)
        self._unwind.clear()
        if self._prof is not None:
            try:
                self._prof.step()  # flush any trailing scheduled state
            except Exception:
                pass
            self._prof.__exit__(None, None, None)
            self._prof = None
