# QK Norm + Partial RoPE Training Fusion — Go/No-Go

Branch: `minimax-h3-profile` (doc-only commit; no code changed).

## 1. Existing traces reused

- eager: `artifacts/e_regional_compile/runs/off_prof/trace.rank0.json` (N=4, step 3, record_shapes, with_stack)
- compiled: `FastVideo-wt-compile/artifacts/minimax_h3_opt_search/regional_compile_e2e_on_prof/trace.rank0.json` (models.student.compile_blocks=true, step 3)
- plus `artifacts/minimax_h3_opt_search/E_attn_preprocess_accounting.md` and `G_small_kernel_buckets.md` as source auditors.

No new GPU/profiler run was needed.

## 2. Eager breakdown (per active step)

| op class | kernels | CUDA time |
|---|---|---|
| Q/K RMSNorm (head_dim=128, [..,56,128]) | 32 | 41.8 ms |
| partial-RoPE pointwise chain (mul/neg/cat/split fp32) | 146+93+28 | 202.5 ms |
| hidden-5376 RMSNorm (whole-model; incl. block norms) | 36 | 34.8 ms |
| layout copies in this region (share of 214.8 ms) incl. dtype casts to fp32 | partial | ~50-90 ms (heavily overlapped with modulation/pack) |

QK Norm+RoPE eager total ≈ **244 ms / 5.55 s step = 4.4%** of step.

## 3. Regional-compile breakdown (step 3, compile on)

| op class | kernels | CUDA time |
|---|---|---|
| Q/K RMSNorm head128 (reduction kernels remain) | 22 | 15.5 ms |
| partial-RoPE pointwise chain | 31 | 3.8 ms (-98.5%) |
| fused-triton norm+pointwise (inductor) | 198 | 107.8 ms (absorbed chains) |

Compiled QK Norm+RoPE residue = **~19.3 ms / 4.78 s step = 0.40%**.

## 4. What compile already fused

`rope_mul` 109.6→2.5 ms, `rope_cat_split` 84.3→1.3 ms, `rope_neg` 8.6→0.02 ms —
the entire partial-rotary pointwise ladder is inside inductor triton kernels.
The norm reduction kernels themselves survive (as they must — reductions).

## 5. Remaining kernel fragmentation after compile

- 15.5 ms of head-128 RMSNorm reductions (inside rope region) + hidden norms —
  cannot be fused away further without changing numerics.
- 3.8 ms of rope pointwise residue.
- fp32 index_add_ (block AdaLN gathers) removed under compile — was 117 ms, now 0; per K audit.
- bf16 final-AdaLN index_add_ 508.8 ms exists in BOTH traces, already handled by the
  independent two-row reduction task (not this one).

## 6. Ceilings vs full step (4.78 s compiled)

| bound | math | % of step |
|---|---|---|
| absolute (100% of 19.3 ms) | 19.3 / 4779 | **0.40%** |
| realistic 50% | 9.6 / 4779 | **0.20%** |
| conservative 25% | 4.8 / 4779 | **
0.10%** |

plus materially freeing NCCL A2A payload (fp32→bf16) is covered by the frozen
dtype-convergence experiment, not this fusion.

## 7. Existing inference fused implementation audit

`fastvideo/models/dits/minimax_h3_fusions/qknorm_rope.py::fused_qknorm_rope`
(Triton, fused RMSNorm + partial 3-axis RoPE in ONE kernel) exists — but has a
hard training gate (`_can_run_minimax_h3_fusion` requires
`not torch.is_grad_enabled()`), i.e. fwd-only with no autograd path. Reusing it
for training requires writing the backward (RMSNorm bwd + partial rotary bwd).

## 8. Mature training implementation availability

- torch.compile/Inductor: already absorbs ≈92% of this — see §4.
- Apex / Transformer Engine: not installed on this env; and their RoPE is 1D
  full-head only, incompatible with H3's 3-axis partial-96/128 without custom
  kernel work.
- Verdict: no off-the-shelf training fused op fits; inductor already did it.

## 9. Verdict

**DROP** — per the task rule: post-compile residue ≈ 0.40% absolute ceiling
< 0.5% threshold → writing a training-capable fused QKNorm+partial-RoPE kernel
today is not worth it. The inference triton fusion stays inference-only.

One-line answer: best realistic full-step upside from a hand-written training
fusion here ≈ **0.1–0.2%**, below the action threshold.
