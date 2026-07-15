#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/l20.env}"

if [[ -r "${CONFIG_PATH}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
  set +a
elif [[ $# -gt 0 ]]; then
  echo "Configuration file not found: ${CONFIG_PATH}" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

: "${MODEL_DIR:?Set MODEL_DIR to a verified local model snapshot directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Local model directory not found: ${MODEL_DIR}" >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM is not installed in the active offline environment." >&2
  exit 2
fi

command=(
  vllm serve "${MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --disable-log-requests
)

if [[ -n "${LLM_API_KEY:-}" ]]; then
  command+=(--api-key "${LLM_API_KEY}")
fi

echo "Starting offline vLLM model=${SERVED_MODEL_NAME} max_model_len=${VLLM_MAX_MODEL_LEN} max_num_seqs=${VLLM_MAX_NUM_SEQS}"
exec "${command[@]}"
