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

: "${LLM_BASE_URL:?Set LLM_BASE_URL to the local OpenAI-compatible /v1 endpoint}"
: "${LLM_MODEL_NAME:?Set LLM_MODEL_NAME to the vLLM served model name}"

export LLM_API_KEY="${LLM_API_KEY:-offline-local}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-256}"
export LLM_CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-4096}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-none}"
export LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-120}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/check_environment.py
"${PYTHON_BIN}" scripts/test_llm_api.py
"${PYTHON_BIN}" scripts/run_synthetic_smoke.py --stage single
