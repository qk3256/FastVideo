#!/usr/bin/env python3
"""Summarize Stage2Evidence output from run_integrated_ab.sh."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


ARMS = ("baseline", "adaln", "compile", "combined")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def summarize(root: Path) -> dict:
    results: list[dict] = []
    for name in ARMS:
        run_dir = root / name
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rows = _read_jsonl(run_dir / "metrics.rank0.jsonl")
        stderr = (run_dir / "stderr.log").read_text(errors="replace") if (run_dir / "stderr.log").exists() else ""
        exit_path = run_dir / "exit_code.txt"
        exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
        steady = [float(row["metrics"]["step_time_sec"]) for row in rows if int(row["step"]) >= 2]
        losses = [float(row["metrics"]["total_loss"]) for row in rows]
        finite = all(bool(row.get("loss_finite")) and math.isfinite(loss) for row, loss in zip(rows, losses))
        result = {
            "run_id": name,
            "git_head": manifest.get("git_head"),
            "compile_blocks": manifest.get("compile_blocks"),
            "final_adaln_two_row_backward": manifest.get("final_adaln_two_row_backward"),
            "exit_code": exit_code,
            "optimizer_steps_observed": [int(row["optimizer_step"]) for row in rows],
            "step_time_sec": [float(row["metrics"]["step_time_sec"]) for row in rows],
            "steady_state_median_sec": statistics.median(steady) if steady else None,
            "loss": losses,
            "loss_finite": finite and bool(rows),
            "weight_changed": bool(rows) and all(bool(row.get("weight_changed")) for row in rows),
            "backward_completed": bool(rows) and all(bool(row.get("backward_completed")) for row in rows),
            "optimizer_step_completed": bool(rows) and all(bool(row.get("optimizer_step_completed")) for row in rows),
            "peak_allocated_bytes": max((int(row["peak_allocated_bytes"]) for row in rows), default=None),
            "peak_reserved_bytes": max((int(row["peak_reserved_bytes"]) for row in rows), default=None),
            "graph_break_log_events": _count(r"Graph break in user code|Graph break from", stderr),
            "recompile_log_events": _count(r"Recompiling function", stderr),
            "oom": "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr,
            "error": exit_code not in (0, None),
        }
        results.append(result)

    baseline = results[0]["steady_state_median_sec"]
    for result in results:
        current = result["steady_state_median_sec"]
        result["speedup_vs_baseline_pct"] = (baseline - current) / baseline * 100.0 if baseline and current else None
    return {"schema_version": 1, "workload": {"transformer_layers": 4, "gpus": "2xA100-SXM4-40GB", "sp_size": 2, "dtype": "bf16"}, "runs": results}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/minimax_h3_integrated_ab")
    summary = summarize(root)
    json_path = root / "integrated_ab_summary.json"
    csv_path = root / "integrated_ab_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    fields = (
        "run_id", "git_head", "compile_blocks", "final_adaln_two_row_backward", "exit_code",
        "steady_state_median_sec", "speedup_vs_baseline_pct", "loss_finite", "weight_changed",
        "peak_allocated_bytes", "peak_reserved_bytes", "graph_break_log_events", "recompile_log_events", "oom", "error",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: run.get(key) for key in fields} for run in summary["runs"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
