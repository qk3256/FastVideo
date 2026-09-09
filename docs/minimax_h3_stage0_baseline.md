# MiniMax-H3 SFT profiling: Stage 0 baseline

## Identity

- upstream baseline commit: `556ac7088e7b4750806d277d31e0db6cd25a5238` (`[refactor] Simplify Wan sampling and tests (#1825)`)
- experiment code commit at doc generation: `52efa8cc031594b1a1094b51702badfbeb6e027f` on branch `minimax-h3-profile`
- remotes: origin `git@github.com:qk3256/FastVideo.git`, upstream `https://github.com/hao-ai-lab/FastVideo.git`
- baseline is ancestor of experiment HEAD: **True**
- tracked-file modifications at generation time: [' M fastvideo-kernel/CMakeLists.txt'] (untracked runtime artifacts: 3)

The per-run `run_manifest.json` stamps the exact `experiment_code_commit` again at
training launch; this document pins the upstream baseline identity.

## Experiment contract

- measured_reduced_model: 4 active main-transformer layers (smoke 1, OOM fallback 2),
  token refiner keeps its original 2 layers; full-parameter SFT, no LoRA.
- full_model_reference (50 layers) is a static-count reference, never a measurement.
- BF16, per-GPU batch 1, FSDP shard=2/replicate=1, SP=2, full activation checkpointing.
- Ramp: N=1 one step, N=4 one step, N=4 three steps (2-layer fallback on OOM).
- Validation, checkpoint saving and trackers are disabled.

## Environment (measured)

- Python 3.11.16 at `/home/why/miniconda3/envs/fastvideo/bin/python`
- PyTorch 2.12.0+cu126, CUDA 12.6, NCCL [2, 29, 3]
- Driver 570.133.20; GPUs: ['NVIDIA A100-SXM4-40GB', 'NVIDIA A100-SXM4-40GB'] ([40442, 40442] MiB)
- World size 2 (single node)

## Model manifest

- path `/home/why/workspace/project/FastVideo/data/models/MiniMax-H3`; checkpoint layers = 50
- safetensors shards = 14, total 66280504216 bytes
- sha256 model_index: `5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2`
- sha256 transformer/config.json: `74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae`
- sha256 transformer index: `ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e`

## Input manifest (real preprocessed parquet)

- `/home/why/workspace/project/FastVideo/data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet` (14683428 bytes, sha256 `f179164af84c6bbb988cf39d303c36c5c0d350363563424729c9d9b028a27ab0`)
- rows: 1, file `1gGQy4nxyUo-Scene-016.mp4`, caption: "A watermelon wearing a helmet is crushed by a hydraulic press, causing it to flatten and burst open."
- video latent shape [24, 37, 48, 84] (finite=True)
- audio latent shape [2, 32, 207]
- text embedding shape [21, 5120]; real text tokens = 21
- geometry: 124 frames -> num_latent_t 37, 1344x768, fps 24.0
- tokens: video 37296, audio 207, text 21;
  packed global 37524, SP-local (SP=2) 18762

## Checks

| check | passed | observed |
|---|---|---|
| baseline_is_ancestor | True | `"merge-base --is-ancestor 556ac708 HEAD -> rc=0"` |
| source_worktree_clean | False | `[" M fastvideo-kernel/CMakeLists.txt", "?? artifacts/", "?? docs/minimax_h3_stage0_baseline.json", "?? docs/minimax_h3_stage0_baseline.md"]` |
| model_manifest_complete | True | `{"checkpoint_num_layers": 50, "safetensors_shard_count": 14, "safetensors_total_bytes": 66280504216}` |
| dataset_manifest_complete | True | `{"rows": 1, "parquet": "/home/why/workspace/project/FastVideo/data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet"}` |
| cuda_available | True | `true` |
| gpu_count_is_2 | True | `2` |
| two_rank_nccl_smoke_passed | True | `"2-rank all-reduce mean=3.0 on both ranks, NCCL (2,29,3), clean exit"` |
| reduced_depth_config_present | True | `"examples/train/configs/profile_minimax_h3_t2va_2xa100.yaml"` |
| offline_inputs_ready | True | `{"model_dir": "/home/why/workspace/project/FastVideo/data/models/MiniMax-H3", "parquet": "/home/why/workspace/project/FastVideo/data/crush-smol_h3_t2va_single_s` |

status: complete — Stage 2 may proceed.
