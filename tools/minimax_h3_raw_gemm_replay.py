#!/usr/bin/env python3
"""Raw GEMM replay from the MiniMax H3 runtime shape corpus (P5).

Standalone single-GPU process. Rebuilds every observed layout per
(scope, op_role) from the runtime hook JSONLs (Fwd X@W^T, Dgrad dY@W,
Wgrad dY^T@X), benchmarks with CUDA events (median/p20/p80, TFLOP/s), and
captures the actual kernel name + cuBLAS family via a short torch profiler
pass. Phase A benches every deduped case (warmup 10 / repeat 30); phase B
re-benches the top-5 cases by observed_count x FLOPs (warmup 20 / repeat 100).
Everything lands under the dedicated raw-GEMM artifact dir, separate from any
full-training-step profiling output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

_ROLES_MAIN = ["Q", "K", "V", "O", "FFN-GateUp", "FFN-Down", "AdaLN-block", "AdaLN-out"]


def load_cases(hooks_dir: Path) -> list[dict]:
    """Rebuild deduped GEMM cases from runtime Fwd/Dgrad records."""
    records = {}
    for rank in (0, 1):
        path = hooks_dir / f"hooks.rank{rank}.jsonl"
        for line in path.read_text().splitlines():
            e = json.loads(line)
            records[(e["module_path"], e["phase"])] = e  # same across steps/ranks; keep last id

    cases: dict[tuple, dict] = {}

    def add_case(role: str, scope: str, kind: str, m: int, n: int, k: int,
                 dtype: str, src_module: str, src_phases: list[str]) -> None:
        key = (scope, role, kind, m, n, k, dtype)
        if key not in cases:
            cases[key] = {
                "scope": scope, "op_role": role, "kind": kind,
                "m": m, "n": n, "k": k, "dtype": dtype,
                "src_module": src_module, "src_phases": sorted(src_phases),
            }
        cases[key]["src_phases"] = sorted(set(cases[key]["src_phases"]) | set(src_phases))

    for (module_path, phase), e in records.items():
        role, scope = e["op_role"], e["scope"]
        if phase == "Fwd" and e["input"] and e["weight"]:
            k = e["input"]["shape"][-1]
            m = math.prod(e["input"]["shape"][:-1])
            n, k_w = e["weight"]["shape"]
            assert k == k_w, (module_path, k, k_w)
            # Each phase carries its own runtime-observed dtype: Fwd inputs
            # may run in fp32 while grads are bf16.
            fwd_dtype = e["input"]["dtype"]
            add_case(role, scope, "Fwd", m, n, k, fwd_dtype, module_path, [phase])
    for (module_path, phase), e in records.items():
        role, scope = e["op_role"], e["scope"]
        if phase not in ("Dgrad", "Wgrad") or not (e["input"] and e["weight"]):
            continue
        grad_dtype = e["input"]["dtype"]
        n_w, k_w = e["weight"]["shape"]
        dy_m, dy_n = e["input"]["shape"][-2] if len(e["input"]["shape"]) >= 2 else 1, e["input"]["shape"][-1]
        m_tokens = math.prod(e["input"]["shape"][:-1])
        if phase == "Dgrad":  # dY[M,N] @ W[N,K] -> [M,K]
            assert dy_n == n_w, (module_path, dy_n, n_w)
            add_case(role, scope, "Dgrad", m_tokens, k_w, n_w, grad_dtype, module_path, [phase])
        else:  # Wgrad: dY^T[N,M] @ X[M,K] -> [N,K]
            assert dy_n == n_w, (module_path, dy_n, n_w)
            add_case(role, scope, "Wgrad", n_w, k_w, m_tokens, grad_dtype, module_path, [phase])
    return list(cases.values())


def _flops(case: dict) -> int:
    return 2 * case["m"] * case["n"] * case["k"]


def bench(case: dict, warmup: int, repeat: int, device: torch.device) -> dict:
    dtype = torch.bfloat16 if case["dtype"] in ("bfloat16", "bf16") else torch.float32
    m, n, k = case["m"], case["n"], case["k"]
    if case["kind"] == "Fwd":
        x = torch.randn(m, k, device=device, dtype=dtype)
        w = torch.randn(n, k, device=device, dtype=dtype)
        run = lambda: x @ w.t()  # noqa: E731
    elif case["kind"] == "Dgrad":
        dy = torch.randn(m, k, device=device, dtype=dtype)  # m x N stored under k slot
        w = torch.randn(k, n, device=device, dtype=dtype)
        run = lambda: dy @ w  # noqa: E731
    else:  # Wgrad: dY.T @ X
        dy = torch.randn(case["k"], case["m"], device=device, dtype=dtype)
        x = torch.randn(case["k"], case["n"], device=device, dtype=dtype)
        run = lambda: dy.t() @ x  # noqa: E731
    torch.cuda.synchronize(device)

    def run_once() -> None:
        # The training wrapper runs the transformer forward under bf16 autocast,
        # so an fp32-recorded forward input still executes as a bf16 GEMM.
        # bf16/gradient operands are unchanged by autocast.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            run()

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
    return {"median_ms": med, "p20_ms": p20, "p80_ms": p80, "tflops": tflops,
            "warmup": warmup, "repeat": repeat,
            "executed_dtype": "bfloat16(autocast)" if case["dtype"] == "float32" else case["dtype"]}


def kernel_name(case: dict, device: torch.device) -> tuple[str, str]:
    """Capture the real kernel + cuBLAS family with a one-off CPU+CUDA profile."""
    dtype = torch.bfloat16 if case["dtype"] in ("bfloat16", "bf16") else torch.float32
    m, n, k = case["m"], case["n"], case["k"]
    if case["kind"] == "Fwd":
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(n, k, device=device, dtype=dtype)
        run = lambda: a @ b.t()  # noqa: E731
    elif case["kind"] == "Dgrad":
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        run = lambda: a @ b  # noqa: E731
    else:
        a = torch.randn(k, m, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        run = lambda: a.t() @ b  # noqa: E731
    run()  # warm/compile outside the profiled region
    torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            run()
        torch.cuda.synchronize(device)
    kernels = sorted(
        (e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: -getattr(e, "self_device_time_total", 0.0))
    name = kernels[0].key if kernels else ""
    lower = name.lower()
    backend = ("cublaslt" if "nvjet" in lower else "cutlass/cublaslt" if "cutlass" in lower
               else "cublas" if ("gemm" in lower or "sgemm" in lower or "h16816" in lower) else "unknown")
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

    phase_a = []
    for case in cases:
        stats = bench(case, warmup=10, repeat=30, device=device)
        phase_a.append({**case, **stats, "flops": _flops(case), "phase": "A"})
        print(f"A {case['scope']}/{case['op_role']}/{case['kind']} m={case['m']} n={case['n']} k={case['k']} "
              f"median={stats['median_ms']:.3f}ms {stats['tflops']:.1f}TF/s")
    with open(args.output_dir / "raw_gemm_phaseA.jsonl", "w") as fh:
        for row in phase_a:
            fh.write(json.dumps(row) + "\n")

    order = sorted(phase_a, key=lambda r: -r["flops"])
    phase_b = []
    for case in order[: args.top]:
        stats = bench(case, warmup=20, repeat=100, device=device)
        kname, backend = kernel_name(case, device)
        phase_b.append({**{k2: v for k2, v in case.items() if k2 != "phase"}, **stats,
                        "phase": "B", "flops": _flops(case), "kernel_name": kname, "backend": backend})
        print(f"B {case['op_role']}/{case['kind']} median={stats['median_ms']:.3f}ms "
              f"{stats['tflops']:.1f}TF/s kernel={kname} backend={backend}")
    with open(args.output_dir / "raw_gemm_phaseB.jsonl", "w") as fh:
        for row in phase_b:
            fh.write(json.dumps(row) + "\n")
    meta = {
        "source_hooks_dir": str(args.hooks_dir),
        "cases": len(cases),
        "top_n_with_kernel": len(phase_b),
        "device": torch.cuda.get_device_name(device),
        "note": "raw GEMM replay isolated from full-step training: no FSDP, no dataloader, no model state",
    }
    (args.output_dir / "raw_gemm_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("raw GEMM replay done")


if __name__ == "__main__":
    main()
