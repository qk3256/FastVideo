# MiniMax H3 SFT profiling — stage 2 + overnight queue summary (post-review v3)

machine: node161 (2x A100-SXM4-40GB, driver 570.133.20)
env: python 3.11.16, torch 2.12.0+cu126, CUDA 12.6, NCCL 2.29.3
branch: minimax-h3-profile (fork qk3256/FastVideo)
upstream baseline: 556ac708; upstream HEAD drifted to a943220c (unrelated Spark preset fix)

## Hotspot summary (rerun_nsys_v2, step-3 capture window, share of CUDA kernel time)
- attention (pytorch_flash fwd+bwd): 41.8%
- bf16 GEMMs (QKV/O/FFN fwd+bwd, ampere_s16816 cublas family): 18.7%
- index/pack (indexFuncLargeIndex bf16+fp32): 12.6%
- NCCL (allgather/reducescatter/alltoall/sendrecv): 17.5%
- NVTX range view (CPU-side durations incl. gaps): minimax_h3.transformer_block.self_attention ≈ 53% — two different measurement frames, both recorded.

## Canonical run dirs (git_head 653040aa for reruns; legacy commit times reconstructed in pack_provenance.json)
- artifacts/minimax_h3_stage2/{s1_l1_step1,s2_l4_step1,s3_l4_steps3}
- artifacts/minimax_h3_overnight/p1_stability_l4_steps7
- artifacts/minimax_h3_rerun_8d99ce72/rerun_s3_l4_steps3 (optimizer_step=1/2/3, finite total_loss, weight_changed)
- rerun_collector_v4 (full shape/stride/dtype/alignment incl. tuple-unpacked outputs and real Wgrad weight.grad)
- rerun_profiler_v2, rerun_nsys_v2
- artifacts/minimax_h3_raw_gemm_v3 (42 cases, 47 bench rows incl. phase-B)

## P6 optimization experiments (both rejected, evidence-based)
- bf16-cast residual streams: REJECTED as analysis artifact (autocast already casts at GEMM entry).
- fused AdaLN gather: numerically faithful, but -34.9% step time from strided chunk-view reads.
  Full A/B on branch minimax-h3-fuse-adaln-gather: profiling/minimax_h3/P6_ADALN_GATHER_CANDIDATE.md there.

## pack contents
shape_corpus.jsonl (151 rows, all inputs/outputs/weights with shape/stride/dtype/alignment;
all 151 join to raw GEMM rows: 42 unique raw cases, RecomputeFwd via raw_equivalent_workload_id),
shape_corpus.schema.json (JSON Schema with $defs.tensor_meta), corpus_join_report.json,
raw_gemm_cases.csv, raw_gemm_results.csv, pytorch_top_ops.csv, nsys_kernel_summary.csv,
nsys_nvtx_summary.csv, run_matrix.csv (git_head + active layers + optimizer steps observed),
traces.index.json (big traces by path+sha256), MANIFEST.sha256.
