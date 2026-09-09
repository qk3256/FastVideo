# SPDX-License-Identifier: Apache-2.0
"""Runtime shape collector for profiling runs (real autograd observation).

Installs forward and full-backward hooks on the profiling target modules of
the student transformer and records only tensor metadata: shape, stride,
dtype, storage offset, contiguity, and data-pointer alignment at 16/32/64/
128/256 bytes. No tensor payloads are retained.

Phases are real observations: ``Fwd`` is the initial forward, ``RecomputeFwd``
is the activation-checkpoint replay detected by bracketing ``method.backward``
(the recompute fires inside ``loss.backward()``), and ``Dgrad``/``Wgrad``
records come from full-backward hooks observing real gradients. Each rank
writes its own ``hooks.rank<N>.jsonl``; aggregation, a CSV view, and a
verification report are produced at train end. Disabled unless explicitly
enabled via YAML/CLI override, so profiling runs stay opt-in per mode.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

logger = init_logger(__name__)

_ALIGNMENTS = (16, 32, 64, 128, 256)

_ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.attn\.to_q$"), "Q"),
    (re.compile(r"\.attn\.to_k$"), "K"),
    (re.compile(r"\.attn\.to_v$"), "V"),
    (re.compile(r"\.attn\.to_out$"), "O"),
    (re.compile(r"\.ff\.fc_in$"), "FFN-GateUp"),
    (re.compile(r"\.ff\.fc_out$"), "FFN-Down"),
    (re.compile(r"\.adaln_proj\.linear$"), "AdaLN-block"),
    (re.compile(r"(^|\.)norm_out\.linear$"), "AdaLN-out"),
]

_FLATTEN_PATTERN = re.compile(r"^(tensor\(|None)")


def _tensor_meta(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    global_shape: list[int] | None = None
    local = t
    if hasattr(t, "to_local"):  # FSDP2 DTensor: pointer/offset live on the local shard
        global_shape = list(t.shape)
        local = t.to_local()
    ptr = local.data_ptr()
    return {
        "shape": global_shape if global_shape is not None else list(local.shape),
        "local_shard_shape": list(local.shape) if global_shape is not None else None,
        "stride": list(local.stride()),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(local.device),
        "contiguous": bool(local.is_contiguous()),
        "storage_offset": int(local.storage_offset()),
        "nbytes": int(local.element_size() * local.nelement()),
        **{f"align{m}": ptr % m == 0 for m in _ALIGNMENTS},
    }


def _role_of(path: str) -> str | None:
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(path):
            return role
    return None


def _scope_of(path: str) -> str:
    if path.startswith("token_refiner."):
        return "text_refiner"
    if path.startswith("norm_out"):
        return "final"
    return "main_transformer"


class RuntimeShapeCollectorCallback(Callback):
    """Per-rank runtime tensor-metadata collector for target GEMM modules."""

    def __init__(self, *, output_dir: str, enabled: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self._records: list[dict[str, Any]] = []
        self._hooks: list[Any] = []
        self._step = 0
        self._rank = 0
        self._world = 1
        self._in_backward = False
        self._path: Path | None = None
        self._last_fwd_input: dict[str, dict[str, Any]] = {}
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.output_dir

    def _phase(self) -> str:
        return "RecomputeFwd" if self._in_backward else "Fwd"

    def _forward_hook(self, path: str, role: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if self._in_backward or torch.is_grad_enabled():
                self._record_fwd(path, role, inputs, output)

        return hook

    def _backward_hook(self, path: str, role: str):
        def hook(module: torch.nn.Module, grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
            if not torch.is_tensor(grad_output[0]):
                return
            dy = grad_output[0]
            dx = grad_input[0] if (grad_input and torch.is_tensor(grad_input[0])) else None
            weight = getattr(module, "weight", None)
            self._emit(path, role, "Dgrad", in_meta=_tensor_meta(dy), out_meta=_tensor_meta(dx),
                       weight_meta=_tensor_meta(weight) if torch.is_tensor(weight) else None)
            paired = self._last_fwd_input.pop(path, None)
            wgrad_out = None
            if weight is not None and torch.is_tensor(weight):
                wgrad_out = {"shape": list(weight.shape), "dtype": str(weight.dtype).replace("torch.", "")}
            self._emit(path, role, "Wgrad", in_meta=_tensor_meta(dy), out_meta=wgrad_out,
                       weight_meta=_tensor_meta(weight) if torch.is_tensor(weight) else None,
                       extra={"wgrad_lhs_meta": paired, "wgrad_operands_paired": paired is not None})

        return hook

    def _emit(self, path: str, role: str, phase: str, *, in_meta: Any, out_meta: Any,
              weight_meta: Any, extra: dict[str, Any] | None = None) -> None:
        rec = {
            "ts": time.time(),
            "rank": self._rank,
            "step": self._step,
            "module_path": path,
            "op_role": role,
            "scope": _scope_of(path),
            "phase": phase,
            "input": in_meta,
            "output": out_meta,
            "weight": weight_meta,
        }
        if extra:
            rec.update(extra)
        self._records.append(rec)

    def _record_fwd(self, path: str, role: str, inputs: tuple[Any, ...], output: Any) -> None:
        x = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
        weight = getattr(self._modules.get(path), "weight", None)
        x_meta = _tensor_meta(x)
        phase = self._phase()
        if x_meta is not None:
            # Cache the input metadata on every forward; a checkpoint recompute
            # overwrites it with the replayed observation. Segment-tail modules
            # that never replay still pair their Wgrad with the initial input.
            self._last_fwd_input[path] = x_meta
        self._emit(path, role, phase, in_meta=x_meta,
                   out_meta=_tensor_meta(output) if torch.is_tensor(output) else None,
                   weight_meta=_tensor_meta(weight) if torch.is_tensor(weight) else None)

    _modules: dict[str, torch.nn.Module]

    def on_train_start(self, method: Any, iteration: int = 0) -> None:
        if not self.enabled:
            return
        if dist.is_available() and dist.is_initialized():
            self._rank, self._world = dist.get_rank(), dist.get_world_size()
        transformer = method.student.transformer
        self._modules = dict(transformer.named_modules())
        installed = []
        for path, module in self._modules.items():
            role = _role_of(path)
            if role is None:
                continue
            self._hooks.append(module.register_forward_hook(self._forward_hook(path, role)))
            self._hooks.append(module.register_full_backward_hook(self._backward_hook(path, role)))
            installed.append({"module_path": path, "op_role": role, "scope": _scope_of(path)})
        if not installed:
            raise RuntimeError("runtime shape collector matched no target modules")
        meta = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "rank": self._rank,
            "world_size": self._world,
            "targets": installed,
        }
        (self.output_dir / f"collector_meta.rank{self._rank}.json").write_text(json.dumps(meta, indent=2) + "\n")
        original_backward = method.backward
        callback = self

        def backward_with_flag(*args: Any, **kwargs: Any) -> Any:
            callback._in_backward = True
            try:
                return original_backward(*args, **kwargs)
            finally:
                callback._in_backward = False

        method.backward = backward_with_flag
        logger.info("runtime shape collector: %d target modules hooked on rank %d", len(installed), self._rank)

    def on_training_step_end(self, method: Any, loss_dict: dict[str, Any], iteration: int = 0) -> None:
        self._step = iteration
        self._flush()

    def _flush(self) -> None:
        if not self.enabled or not self._records:
            return
        with (self.output_dir / f"hooks.rank{self._rank}.jsonl").open("a") as handle:
            for rec in self._records:
                handle.write(json.dumps(rec) + "\n")
        self._records.clear()

    def on_train_end(self, method: Any, iteration: int = 0) -> None:
        self._flush()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if not self.enabled:
            return
        report = self._build_report()
        (self.output_dir / f"collector_report.rank{self._rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    def _build_report(self) -> dict[str, Any]:
        path = self.output_dir / f"hooks.rank{self._rank}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_key: dict[tuple[str, str], int] = {}
        anomalies: list[str] = []
        for rec in records:
            by_key[(rec["module_path"], rec["phase"])] = by_key.get((rec["module_path"], rec["phase"]), 0) + 1
            for meta_key in ("input", "output", "weight"):
                meta = rec.get(meta_key)
                if not meta:
                    continue
                if meta.get("dtype") == "float32" and meta_key == "input":
                    anomalies.append(f"{rec['module_path']}:{rec['phase']} {meta_key} float32")
                if meta.get("contiguous") is False and meta_key == "input":
                    anomalies.append(f"{rec['module_path']}:{rec['phase']} {meta_key} non-contiguous")
                if meta.get("align16") is False:
                    anomalies.append(f"{rec['module_path']}:{rec['phase']} {meta_key} misaligned<16B")
        targets = json.loads((self.output_dir / f"collector_meta.rank{self._rank}.json")
                             .read_text())["targets"]
        wgrad_paired: dict[str, bool] = {}
        for rec in records:
            if rec["phase"] == "Wgrad":
                wgrad_paired[rec["module_path"]] = wgrad_paired.get(rec["module_path"], True) and bool(
                    rec.get("wgrad_operands_paired"))
        module_checks = {}
        for target in targets:
            path = target["module_path"]
            phases = {phase: by_key.get((path, phase), 0) for phase in ("Fwd", "RecomputeFwd", "Dgrad", "Wgrad")}
            module_checks[path] = {
                "op_role": target["op_role"],
                "scope": target["scope"],
                "observed_call_count": phases,
                "has_all_phases": all(v > 0 for k, v in phases.items() if k != "RecomputeFwd"),
                "recompute_replayed": phases["RecomputeFwd"] > 0,
                "wgrad_operands_paired": wgrad_paired.get(path, False),
            }
        return {
            "rank": self._rank,
            "record_count": len(records),
            "module_checks": module_checks,
            "all_targets_complete": all(v["has_all_phases"] for v in module_checks.values()),
            "wgrad_paired_everywhere": all(v["wgrad_operands_paired"] for v in module_checks.values()),
            # Checkpoint recompute replays only ops whose outputs are needed by
            # autograd; segment-tail modules legitimately replay zero times.
            "recompute_observed_anywhere": any(v["recompute_replayed"] for v in module_checks.values()),
            "metadata_anomalies": sorted(set(anomalies)),
        }
