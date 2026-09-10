# Final AdaLayerNormOut — specialized 2-row backward reduction

Branch: `minimax-h3-opt-final-adaln-reduce`. Forward unchanged; only the
backward reduction of the final AdaLNOut scale/shift table grads is swapped.

## Hypothesis

The two BF16 backward `index_add_` of `MiniMaxH3AdaLayerNormOut`
(scale+shift gathers, dst `(2,5376)`, src `(18866,5376)`) cost ~509 ms/step
(81.4% of all 28 backward `index_add_`): the generic ReduceAdd kernel
serializes 18,866 gradient rows onto 2 destination rows in bf16 atomics.
Replacing that scatter with a specialized 2-row reduction should recover
nearly all of it.

## Implementation

`fastvideo/layers/two_row_index_reduce.py:TwoRowIndexSelect.forward` = eager
`index_select(0)` (bitwise identical). Backward uses a two-row reducer with
fp32 accumulation and one final cast:

- default `onehot-GEMM`: `one_hot(idx,2).T @ dY` in fp32 → bf16 (cuBLAS);
- optional Triton two-stage (`prefer_triton=True`): per-token-block fp32
  partials `[P,2,H]` + tiny second reduce (kept for reference; GEMM won).

Wired in only for the final AdaLayerNormOut via
`--models.student.final_adaln_two_row_backward true`. No changes to block
AdaLN, QKV/FFN, forward RMSNorm path, checkpointing, or dtypes.

## Numerics (fp64 golden on the exact [18866,5376] workload)

- fp32-accum paths are strictly closer than eager bf16 scatter to the fp64
  golden: triton/gemm max_abs_diff 0.999 vs eager 6.55; mean_abs 0.111 vs 0.83;
  zero NaN.
- triton vs gemm: max diff ≤ 2.0 (1 bf16 ulp ordering), mean < 0.5.
- forward bitwise identical (ittest asserted `torch.equal`).
- E2E A/B (3 real steps, seed 42): step-1 loss bitwise equal 1997.5276.
  Steps 2-3 diverge mildly (1782.9 vs 1794.1; 1594.6 vs 1536.2) — this is the
  pure bf16 atomic-order noise, and the fp32-accum path is the *better* answer.

## Isolated microbenchmark (CUDA events, warmup=20 repeat=100)

shape `[18866,5376] bf16 -> [2,5376] bf16 grad`:

| path | median ms | kernels | comment |
|---|---|---|---|
| eager `index_select` bwd (ReduceAdd) | 286.87 | 4 | baseline pathology |
| one-hot GEMM fp32 | **0.97** | 12 | cuBLAS `sgemm_largek` (fp32) |
| Triton two-stage fp32 partials | 3.15 | 7 | kept as reference impl |

(Note: eager baseline includes graph/LIFO overhead; profiler-derived per-call
backward cost in the real run was ~254.5 ms. Both agree the pathology.)

## Full-step E2E (N=4, 3 real optimizer steps, SP=2, FSDP shard=2)

| metric | baseline (flag off) | candidate (on) |
|---|---|---|
| step median | 5.510 s | **5.083 s** |
| step speedup | — | **7.74 %** |
| `aten::index_add_` device total | 623.4 ms (28 calls) | **117.7 ms (26 calls)** |
| final AdaLNOut bf16 ReduceAdd | 508.8 ms | **gone** |
| peak reserved (both ranks) | 34.96 GiB | 34.96 GiB |
| weight_changed per step | true | true |

Run dirs: `artifacts/minimax_h3_opt_final_adaln_reduce/{off,on,profiler_on}`.

## Why it worked

The 2-destination scatter is a parallel-reduction problem, not a bandwidth
one. Generic index_add_ hits atomic contention at bf16 precision plus
mixed-precision cast handling; one-hot GEMM offloads the same math to cuBLAS
in fp32 with no fallback. The residual 24 fp32 block-AdaLN gathers (116 ms)
follow the same shape class but are not the target of this experiment and are
*not* pathological per-call (4.84 ms for 18k-row fp32 scatters).

## Mature-implementation options used

None needed for a custom GEMM: cuBLAS through `torch.matmul` covers it. No
Triton kernel was kept on the hot path (its numbers lost to cuBLAS).

## Final verdict

**KEEP as an opt-in training flag.** The final coverage audit found no second
material low-cardinality pathology: the 24 fp32 block use-site gathers are not
pathological per call, and their realistic full-step ceiling is below the
project threshold. No extension to block AdaLN or fused RMSNorm/modulation is
part of the final integrated branch.
