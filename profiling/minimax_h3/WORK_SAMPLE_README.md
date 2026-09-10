# MiniMax-H3 SFT Profiling & Optimization Work Sample

## Task

Profile MiniMax-H3 SFT QKV/O, FFN gate+up/down, and AdaLN forward, Dgrad,
and Wgrad; collect machine-readable shapes/strides/alignment/dtypes/call counts
and kernel backends; separate raw GEMM from full-step evidence; then validate
candidate optimizations with real optimizer-step A/B runs.

## Repro workload

- MiniMax-H3 with 4 Transformer layers (checkpoint depth is 50)
- 2 x NVIDIA A100-SXM4-40GB
- sequence parallel size 2, bf16, one real sample, 3 optimizer steps
- config: [`examples/train/configs/profile_minimax_h3_t2va_2xa100.yaml`](../../examples/train/configs/profile_minimax_h3_t2va_2xa100.yaml)

Run all four integrated arms after activating the FastVideo environment and
placing the model/data at the paths in that config:

```bash
bash profiling/minimax_h3/run_integrated_ab.sh
```

The script explicitly sets both optimization flags for every arm and writes
raw per-rank Stage2Evidence records plus derived JSON/CSV summaries under
`artifacts/minimax_h3_integrated_ab/`.

## Profiling evidence

- runtime corpus: [`shape_corpus.jsonl`](shape_corpus.jsonl) and [`shape_corpus.schema.json`](shape_corpus.schema.json)
- raw GEMM: [`raw_gemm_cases.csv`](raw_gemm_cases.csv) and [`raw_gemm_results.csv`](raw_gemm_results.csv)
- full step: [`pytorch_top_ops.csv`](pytorch_top_ops.csv), [`nsys_kernel_summary.csv`](nsys_kernel_summary.csv), and [`nsys_nvtx_summary.csv`](nsys_nvtx_summary.csv)
- provenance: [`run_matrix.csv`](run_matrix.csv), [`pack_provenance.json`](pack_provenance.json), and [`MANIFEST.sha256`](MANIFEST.sha256)
- decisions: [`FINAL_RANKING.md`](FINAL_RANKING.md)

Large raw traces are intentionally omitted from Git; [`traces.index.json`](traces.index.json)
records their local paths and hashes, while derived summaries are tracked.

## Final KEEP optimizations

1. Regional `torch.compile` on repeated Transformer blocks. It targets
   pointwise/cast/RoPE/norm/cat and dispatch fragmentation. Opt in with
   `models.student.compile_blocks=true`; default is `false`.
2. Final AdaLNOut two-row backward reduction. Forward remains `index_select`;
   backward replaces the contended 18,866-to-2 bf16 scatter with a cuBLAS
   one-hot GEMM using fp32 accumulation. Opt in with
   `models.student.final_adaln_two_row_backward=true`; default is `false`.

The integrated reviewer branch is `minimax-h3-work-sample`. The optimization
code tested below is commit `3979c853` (tree `9d5b57e1`); the isolated A100
validation clone applied the same patches at commit `c3f8c36a` and had the
identical tree hash.

## Results

Fresh integrated run on 2026-09-10 UTC. Steady state is the median of steps 2
and 3; step 1 includes warmup/Inductor build. Full records are in
[`integrated_ab_results.json`](integrated_ab_results.json) and
[`integrated_ab_results.csv`](integrated_ab_results.csv).

| arm | compile | AdaLN | median s/step | vs baseline | peak allocated | peak reserved | graph breaks | recompiles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | off | off | 5.426 | — | 31.51 GiB | 34.96 GiB | 0 | 0 |
| AdaLN | off | on | 4.963 | -8.52% | 31.51 GiB | 34.96 GiB | 0 | 0 |
| compile | on | off | 4.756 | -12.35% | 27.06 GiB | 35.00 GiB | 0 | 1/rank |
| combined | on | on | 4.218 | -22.27% (1.286x) | 27.06 GiB | 35.00 GiB | 0 | 1/rank |

All four arms exited 0, completed 3/3 backward and optimizer steps, had finite
losses, changed a tracked weight, and had no OOM. The compile recompile was the
known step-1-to-step-2 `hidden_states` stride guard, once on each rank. The
fresh medians agree with the historical 5.484/5.022/4.845/4.293 s result set;
no workload was changed to force exact agreement.

The two mechanisms stacked well on this N=4 workload; this is an empirical
interaction result, not a claim of complete mathematical orthogonality.
Final-AdaLN backward is not bitwise equivalent to the eager bf16 atomic
reduction: it deliberately accumulates in fp32 and is closer to an fp64
golden result. Forward values are unchanged.

## Negative results and archive branches

See [`ATTENTION_FINAL_DECISION.md`](ATTENTION_FINAL_DECISION.md) and
[`COVERAGE_MATRIX.md`](COVERAGE_MATRIX.md) for the closed KEEP/DROP evidence.

| branch | status |
|---|---|
| `minimax-h3-opt-regional-compile` | KEEP / standalone review |
| `minimax-h3-opt-final-adaln-reduce` | KEEP / standalone review |
| `minimax-h3-opt-fused-adamw` | DROP as performance optimization / archival |
| `minimax-h3-fuse-adaln-gather` | REJECTED / negative result |
| `minimax-h3-attr-index-add` | ATTRIBUTION ONLY |

The archival branches are intentionally not merged into the integrated branch.

## Scope limitation

The full 50-layer MiniMax-H3 SFT was not run on 2 x 40GB A100. Only N=4 is
measured. Regional compile acts on every repeated Transformer block, but the
full-model compute/communication/fixed-cost mix may differ, so its percentage
cannot be extrapolated directly. Final AdaLN primarily changes the final
output AdaLN and does not scale linearly with depth, so its full-model relative
gain is expected to be smaller. The combined percentage above is N=4 only.
