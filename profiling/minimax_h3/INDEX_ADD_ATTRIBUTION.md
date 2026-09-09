# aten::index_add_ attribution — MiniMax H3 N=4 SFT profiled step 3

Branch: `minimax-h3-attr-index-add` (temporary; pinned on baseline 038902a3 +
one instrumentation commit adding a `with_stack` switch to the profiler
callback). No modeling code changed on this branch's measured path; the run
used the stock fallback AdaLN path (fuse flags off).

Trace: `artifacts/minimax_h3_attribution/profiler_stack_l4_steps3/trace.rank0.json`
(rank 0, profiled step 3, record_shapes=True).
Parser + machine output: `artifacts/minimax_h3_attribution/{index_add_attribution.py,
attribution_rank0.json, attribution_rank0.md}`.

## Method and limitations (honest)

- Backward `aten::index_add_` events in this torch 2.12 trace carry **no**
  `Fwd thread id` + `Sequence number` link, and with_stack produced **zero**
  `Call stack` args (0/28). Forward NVTX ranges cannot cover autograd backward,
  so we do not use forward-range inclusive time as backward time.
- Instead we bucket by the exact runtime fingerprint (dtype x self/src/index
  shapes), which in this workload maps 1:1 to source sites, and we
  cross-verify with `key_averages()` self-device time by kernel count/dtype
  (aten::index_add_ => indexFuncLargeIndex ReduceAdd kernels).
- Kernel events are joined to ops by `External id`; unit of kineto chrome-trace
  `dur` is microseconds (a first pass of this report mislabeled them).

## Source mapping (verified against branch source)

| site | fwd call pattern |
|---|---|
| block AdaLN, lines 543/560/563/566 | scale/shift/gate x {msa,mlp} .index_select(0, adaln_indices) |
| final MiniMaxH3AdaLayerNormOut:462 | scale/shift .index_select(0, timestep_indices) |
| output modality select :960-961 | video_output/audio_output .index_select(1, ..) |

## Attribution table (28 backward index_add_, profiled step 3, rank 0)

| category | calls | dtype | self (dst) | index | src (grad rows) | CUDA time | share |
|---|---|---|---|---|---|---|---|
| 4 blocks x 6 AdaLN use sites | 24 | fp32 | (6, 5376) | (18866,) | (18866, 5376) fp32 grads | 116.2 ms | 18.6% |
| final AdaLNOut scale+shift | 2 | bf16 | (2, 5376) | (18866,) | (18866, 5376) bf16 grads | 508.8 ms | 81.4% |
| video output select | 1 | bf16 | (1, 37731, 96) | (37296,) | (1, 37296, 96) | 0.13 ms | 0.02% |
| audio output select | 1 | bf16 | (1, 37731, 32) | (414,) | (1, 414, 32) | 0.03 ms | 0.01% |
| **total** | 28 | | | | | **625.2 ms** | 100% |

Cross-check with `key_averages`: aten::index_add_ = 28 calls, self device
625,189 us; bf16 ReduceAdd kernel = 4 calls / 508,988 us; fp32 ReduceAdd
kernel = 24 calls / 116,201 us. Consistent.

## Hypotheses

- H1 (24 FP32 = 4 blocks x 6 AdaLN gathers): CONFIRMED (fp32 because AdaLN
  tables are cast to the fp32 residual stream before gathering).
- H2 (4 BF16 = 2 final AdaLNOut + 2 modality output select): CONFIRMED.
- H3 (final AdaLNOut dominates the BF16 4): CONFIRMED with numbers:
  final AdaLNOut = 508.8 ms of 509.0 ms BF16 time (99.96%); the two
  output-select gathers cost 0.163 ms combined. (The two selects' backward
  also partially surfaces as aten::index_copy_/index_fill_ at 1.35/0.90 ms.)

## Interpretation

- Final AdaLNOut's two bf16 gathers pay 254 ms each because a 2-row-wide
  bf16 ReduceAdd scatter over 18,866 gradient rows is atomics-serialized
  (vs ~4.84 ms per fp32 6-row block gather). Kernel count is tiny; cost is
  atomicity + dtype.
- Step context: aten::index_add_ = 625.2 ms of a ~5.4 s step (~11.6%).

## Optimization decision (measurement-driven; NOT implemented here)

SHOULD be pursued: a training-capable fused RMSNorm+indexed-modulation for the
final AdaLNOut (bf16-in/fp32-accumulate scatter, or convert the final modulate
to fp32 like the block path), est. up to ~9% step time. Prerequisites before
implementation: a numerics plan for bf16 atomic accumulation, and a check of how
the 50-layer full model's nsys shape scales this share up (same absolute gather
cost per 50 active layers would amortize away; at 4 layers it is concentrated).

Superseded note: an earlier nsys-only estimate put all index kernels at
~12% of kernel time (mixing forward gather + backward scatter across both
ranks). This report supersedes it for the backward side.
