#!/usr/bin/env python3
"""Attribute aten::index_add_ calls in a with_stack PyTorch profiler trace.

Chain per backward index_add_:
  backward cpu_op.args["Fwd thread id" / "Sequence number"]
    -> forward cpu_op (same seq+thread) -> its "Call stack" (source lines)
  backward cpu_op.args["External id"] -> kernel events (same External id)
    -> CUDA time per backward index_add_

Report buckets by source file:lineno + dtype + shapes.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def norm_id(v):
    return str(v) if v is not None else None


def main(trace_path: Path, out_json: Path, out_md: Path) -> None:
    d = json.load(open(trace_path))
    evs = d["traceEvents"]
    cpu_ops = [e for e in evs if e.get("cat") == "cpu_op"]
    kernels = [e for e in evs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]

    # fwd lookup: (Fwd thread id of a FWD op == its own tid) + Sequence number
    fwd_by_seq = {}
    for e in cpu_ops:
        a = e.get("args", {})
        seq = norm_id(a.get("Sequence number"))
        ftid = norm_id(a.get("Fwd thread id"))
        if seq is not None and ftid is not None:
            fwd_by_seq[(ftid, seq)] = e  # latest occurrence wins (later overwrites)

    kernel_time_by_extid = defaultdict(float)
    kernel_names_by_extid = defaultdict(set)
    for e in kernels:
        ext = norm_id(e.get("args", {}).get("External id"))
        if ext is not None:
            kernel_time_by_extid[ext] += e.get("dur", 0.0)
            kernel_names_by_extid[ext].add(e.get("name", ""))

    groups = defaultdict(lambda: {"count": 0, "cuda_ns": 0.0, "dtype": set(), "self_shape": set(),
                                  "src_shape": set(), "idx_shape": set(), "stacks": set()})
    unattributed = defaultdict(lambda: {"count": 0, "cuda_ns": 0.0, "dtype": set(),
                                        "self_shape": set(), "src_shape": set(), "idx_shape": set()})

    for e in cpu_ops:
        if e.get("name") != "aten::index_add_":
            continue
        a = e.get("args", {})
        dims = a.get("Input Dims") or []
        self_shape = tuple(dims[0]) if len(dims) > 0 and dims[0] else None
        idx_shape = tuple(dims[2]) if len(dims) > 2 and dims[2] else None
        src_shape = tuple(dims[3]) if len(dims) > 3 and dims[3] else None
        dtype = (a.get("Input type") or ["?"])[0]
        ext = norm_id(a.get("External id"))
        cuda_ns = kernel_time_by_extid.get(ext, 0.0)
        knames = sorted(kernel_names_by_extid.get(ext, set()))

        ftid = norm_id(a.get("Fwd thread id"))
        seq = norm_id(a.get("Sequence number"))
        fwd = fwd_by_seq.get((ftid, seq)) if (ftid is not None and seq is not None) else None
        if fwd is not None:
            stack = fwd.get("args", {}).get("Call stack", "")
            h3_lines = [f for f in stack.split(";") if "minimax_h3" in f]
            src = h3_lines[-1].strip() if h3_lines else (stack.split(";")[-1].strip() if stack else "?")
            fwd_name = fwd.get("name", "?")
            label = f"{fwd_name} @ {src}"
            g = groups[label]
            g["stacks"].add(stack)
        else:
            label = f"no-fwd-link self={self_shape} src={src_shape}"
            g = unattributed[label]
        g["count"] += 1
        g["cuda_ns"] += cuda_ns
        g["dtype"].add(dtype)
        if self_shape:
            g["self_shape"].add(str(self_shape))
        if src_shape:
            g["src_shape"].add(str(src_shape))
        if idx_shape:
            g["idx_shape"].add(str(idx_shape))

    total_cuda = sum(g["cuda_ns"] for g in groups.values()) + sum(g["cuda_ns"] for g in unattributed.values())

    def fmt(g):
        return {
            "count": g["count"],
            "dtype": sorted(g["dtype"]),
            "self_shape": sorted(g["self_shape"]),
            "src_shape": sorted(g["src_shape"]),
            "idx_shape": sorted(g["idx_shape"]),
            "cuda_time_ms": round(g["cuda_ns"] / 1e3, 3),
            "pct_of_index_add_cuda": round(100.0 * g["cuda_ns"] / total_cuda, 2) if total_cuda else 0.0,
        }

    report = {
        "trace": trace_path.name,
        "total_index_add_calls": sum(g["count"] for g in groups.values()) + sum(g["count"] for g in unattributed.values()),
        "total_index_add_cuda_ms": round(total_cuda / 1e3, 3),
        "attributed": {k: fmt(v) for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["cuda_ns"])},
        "unattributed": {k: fmt(v) for k, v in unattributed.items()},
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    with out_md.open("w") as fh:
        fh.write(f"# aten::index_add_ attribution ({trace_path.name})\n\n")
        fh.write(f"total calls: {report['total_index_add_calls']}, total CUDA: {report['total_index_add_cuda_ms']} ms\n\n")
        fh.write("## attributed (via fwd link)\n\n")
        fh.write("| forward op @ source | count | dtype | self shape | idx shape | src shape | cuda ms | share |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["cuda_ns"]):
            f = fmt(v)
            fh.write(f"| {k} | {f['count']} | {'/'.join(f['dtype'])} | {f['self_shape']} | {f['idx_shape']} | {f['src_shape']} | {f['cuda_time_ms']} | {f['pct_of_index_add_cuda']}% |\n")
        fh.write("\n## unattributed (no fwd link)\n\n| signature | count | dtype | self shape | idx shape | cuda ms | share |\n|---|---|---|---|---|---|---|\n")
        for k, v in unattributed.items():
            f = fmt(v)
            fh.write(f"| {k} | {f['count']} | {'/'.join(f['dtype'])} | {f['self_shape']} | {f['idx_shape']} | {f['cuda_time_ms']} | {f['pct_of_index_add_cuda']}% |\n")
    print(json.dumps({"calls": report["total_index_add_calls"], "cuda_ms": report["total_index_add_cuda_ms"],
                      "attributed_groups": len(groups), "unattributed_groups": len(unattributed)}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
