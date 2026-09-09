# MiniMax H3 SFT profiling — stage 2 + overnight queue summary

machine: node161 (2x A100-SXM4-40GB, driver 570.133.20)
env: python 3.11.16, torch 2.12.0+cu126, CUDA 12.6, NCCL 2.29.3
branch: minimax-h3-profile (fork qk3256/FastVideo), upstream baseline 556ac708

## Identity & contract
- upstream_baseline_commit = 556ac7088e7b4750806d277d31e0db6cd25a5238 (ancestor of all experiment commits)
- measured_reduced_model = N=4 main layers + original 2-layer token refiner, full-parameter SFT
- full_model_reference = 50 layers, static counts only (never measured)
- BF16 weights, FSDP shard=2/replicate=1, SP=2, full activation checkpointing, batch=1/doc=1
- validation/checkpoint/tracker all off (trackers:["none"] opt-out token)

## Stage 2 (real training closure, real preprocessed data)
- data: data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet (1 row,
  freshly preprocessed from wlsaidhi/crush-smol-merged videos/1gGQy4nxyUo-Scene-016.mp4
  using the real H3 video VAE / audio VAE / Qwen3-VL text encoder with CPU offload)
- s1_l1_step1  PASS: N=1 load+fwd+1 step; loss finite; weight_changed true
- s2_l4_step1  PASS: N=4 1 step; peak rsv 34.96 GiB/rank
- s3_l4_steps3 PASS: N=4 3 steps; loss 1997.5 -> 1781.5 -> 1590.1;
  step ~5.3-5.5s; peak alloc 31.51 GiB, peak rsv 34.96 GiB stable; weight checksum changed
- note: default allocator OOMed at step 3 due to fragmentation; PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True fixed it, no input shrink, no layer fallback needed

## P1 stability baseline (4 layers, no profiler/hooks)
- 7 steps: steady-state 5.28/5.43/5.52/5.67 s (min/median/mean/p95); loss 1997.5 -> 260.8

## P2 runtime shape collector (separate run)
- 41 target modules x phases, 906 per-rank records; Fwd/Dgrad/Wgrad all observed;
  checkpoint recompute separated; rank0/rank1 shapes identical (SP=2); inputs 16..256B aligned
- finding: residual-stream inputs are fp32 pre-autocast; actual GEMMs execute bf16
  (see P5 correction)

## P3 torch profiler (separate run)
- schedule wait=1/warmup=1/active=1; step 3 traced; per-rank chrome traces (~15.8 MB each),
  cpu/cuda self-time key averages + top200 CSV exports

## P4 nsys (separate run, capture-range on step 3)
- trace.nsys-rep 2.9 MB + sqlite 9.1 MB; CUDA kernel summary + API summary + NVTX summary
- minimax_h3 NVTX ranges visible (self_attention 52.5% of NVTX time; per-block ranges)
- GPU kernel: flash_bwd 25.6%, flash_fwd 18.5% (SDPA flash), bf16 GEMMs ~20%,
  index/pack (bf16 indexFuncLargeIndex) 11.1%, NCCL ~10%
- one teardown hang observed on the NVTX-enabled rerun; killed our own processes,
  retried cleanly

## P5 raw GEMM replay (standalone, separate dir artifacts/minimax_h3_raw_gemm_v2)
- 42 deduped cases from runtime corpus; warmup 10/repeat 30; top-5 FLOPs x count re-benched
  warmup 20/repeat 100 with kernel-name capture
- faithful bf16-autocast replay: main GEMMs ~210-224 TF/s
- v1 without autocast showed fp32 SGEMM 17.7 TF/s — analysis artifact, retracted

## P6 decisions
- candidate "cast norm outputs to bf16 before GEMMs": REJECTED as artifact (autocast
  already casts; nsys proves bf16 kernels). See commit history note; no model code merged.
- layer sweep 5/6/7: SKIPPED by rule — N=4 peak reserved 34.96 GiB exceeds the 34 GiB
  continue gate; predicted N=5 ≈ 37.9 GiB > 35 GiB stop threshold. primary_profile_layers = 4.
- candidate "fused AdaLN row gather": tested on branch minimax-h3-fuse-adaln-gather,
  numerically faithful (step-1 loss bitwise equal) but 34.9% SLOWER per step (strided
  chunk-view reads) — REJECTED; see profiling/minimax_h3/P6_ADALN_GATHER_CANDIDATE.md
  on that branch.

## artifact index
- artifacts/minimax_h3_stage2/{s1_l1_step1,s2_l4_step1,s3_l4_steps3}/ — run dirs with
  command.sh, resolved_config.yaml, run_manifest.json, stdout/stderr, metrics.rank*.jsonl,
  evidence summaries, exit codes; s3 also has the real-parquet static shape corpus CSV+JSON
- artifacts/minimax_h3_overnight/p1_stability_l4_steps7/
- artifacts/minimax_h3_overnight/p2_runtime_collector_l4_steps3/ (hooks.rank*.jsonl,
  collector_report/ aggregated_hooks.csv + summary)
- artifacts/minimax_h3_overnight/p3_torch_profiler_l4_steps3/
- artifacts/minimax_h3_overnight/p4_nsys_l4_steps3/
- artifacts/minimax_h3_raw_gemm_v2/ (phaseA/phaseB jsonl + meta)
- docs/minimax_h3_stage0_baseline.{json,md} — rebuilt stage-0 baseline (committed)

## commits on minimax-h3-profile
bfbf51c3 P0 loader forwarding fix
87f6b2a5 stage2 evidence callback
52efa8cc preprocess text-encoder CPU offload
87e63482 stage0 docs
81a174da + 834bd64c tracker off switch
a24ca561 shape-collector parquet text tokens
9b53493a runtime shape collector callback
7c4e24c2 torch profiler callback
396d1606 nsys capture bracket

## Review remediation addendum (commits ac82b3ad, b4d17abd, ef11cf14, 389e08e5)
- stage-0 audio token accounting fixed: 414 audio rows (2 stereo ch x 207), global 37731,
  SP-padded 37732, SP-local 18866; runtime/main/static/raw GEMM already used M=18866.
- Wgrad pairing: collector now caches input metadata on every observed forward;
  rerun at N=4 shows wgrad_operands_paired=true on all 41 targets (both ranks).
- key filter keeps illegal block indices visible (strict load then raises unexpected keys).
- evidence entries: trainer_step vs optimizer_step from optimizer state; loss_finite
  requires an explicit finite total_loss.
- upstream/main at report time: a943220c (one unrelated Spark preset commit past baseline).
- kernel CMakeLists overlay pinned by diff sha256 3666f9aa89220b50 (build-only, unused by
  the TORCH_SDPA training path).
- this pack: shape_corpus.jsonl (151 rows, runtime-provenanced), raw_gemm_cases/results.csv,
  pytorch_top_ops.csv, nsys_kernel_summary.csv, nsys_nvtx_summary.csv, run_matrix.csv,
  traces.index.json (big traces by path+sha256 only), MANIFEST.sha256.
