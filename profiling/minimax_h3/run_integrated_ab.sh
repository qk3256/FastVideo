#!/usr/bin/env bash
# Reproduce the four-way MiniMax-H3 N=4 integrated-optimization A/B.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="examples/train/configs/profile_minimax_h3_t2va_2xa100.yaml"
OUTPUT_ROOT=${1:-"${REPO_ROOT}/artifacts/minimax_h3_integrated_ab"}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29840}
LOCK_DIR=${H3_GPU_LOCK_DIR:-/tmp/h3_gpu_lock}

cd "${REPO_ROOT}"
test -f "${CONFIG}"
test -d data/models/MiniMax-H3
test -d data/crush-smol_h3_t2va_single_sample_preprocessed
mkdir -p "${OUTPUT_ROOT}"

while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
  echo "waiting for ${LOCK_DIR} ..." >&2
  sleep 15
done
cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ANY_FAILED=0

run_arm() { # name compile_blocks final_adaln_two_row_backward port
  local name=$1 compile_blocks=$2 final_adaln=$3 port=$4
  local run_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${run_dir}"

  python - "${run_dir}" "${name}" "${compile_blocks}" "${final_adaln}" "${CONFIG}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

run_dir, name, compile_blocks, final_adaln, config = sys.argv[1:]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
manifest = {
    "run_id": name,
    "git_head": commit,
    "config_path": config,
    "config_sha256": hashlib.sha256(pathlib.Path(config).read_bytes()).hexdigest(),
    "started_at": datetime.now(timezone.utc).isoformat(),
    "workload": {"transformer_layers": 4, "world_size": 2, "sp_size": 2, "dtype": "bf16", "optimizer_steps": 3},
    "compile_blocks": compile_blocks == "true",
    "final_adaln_two_row_backward": final_adaln == "true",
}
pathlib.Path(run_dir, "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

  local -a command=(
    python -m torch.distributed.run --nproc_per_node=2 --master_port="${port}"
    -m fastvideo.train.entrypoint.train --config "${CONFIG}"
    --models.student.num_transformer_layers 4
    --models.student.compile_blocks "${compile_blocks}"
    --models.student.final_adaln_two_row_backward "${final_adaln}"
    --training.loop.max_train_steps 3
    --callbacks.stage2_evidence.output_dir "${run_dir}"
    --callbacks.runtime_shape_collector.enabled false
    --callbacks.torch_profiler.enabled false
    --callbacks.nsys_capture.enabled false
  )
  printf '%q ' "${command[@]}" > "${run_dir}/command.sh"
  printf '\n' >> "${run_dir}/command.sh"

  local exit_code
  set +e
  if [[ "${compile_blocks}" == "true" ]]; then
    env PYTHONPATH="${REPO_ROOT}" TORCH_LOGS=graph_breaks,recompiles timeout 660 "${command[@]}" \
      > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log"
  else
    env PYTHONPATH="${REPO_ROOT}" timeout 660 "${command[@]}" \
      > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log"
  fi
  exit_code=$?
  set -e
  printf '%s\n' "${exit_code}" > "${run_dir}/exit_code.txt"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "${name} failed with exit ${exit_code}; see ${run_dir}/stderr.log" >&2
    ANY_FAILED=1
  fi
}

run_arm baseline false false $((MASTER_PORT_BASE + 0))
run_arm adaln    false true  $((MASTER_PORT_BASE + 1))
run_arm compile  true  false $((MASTER_PORT_BASE + 2))
run_arm combined true  true  $((MASTER_PORT_BASE + 3))

python profiling/minimax_h3/summarize_integrated_ab.py "${OUTPUT_ROOT}"
if [[ "${ANY_FAILED}" -ne 0 ]]; then
  exit 1
fi
