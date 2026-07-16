#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONDA_ROOT="${CONDA_ROOT:-/opt/miniconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/opt/miniconda3/envs/pytorch}"
OUTPUT_DIR="${L20_SMOKE_OUTPUT_DIR:-$ROOT_DIR/artifacts/l20_smoke}"

: "${MODEL_PATH:?Set MODEL_PATH to the verified local Transformers model directory}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -r "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "Conda activation script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 2
fi
if [[ ! -x "$CONDA_ENV_PATH/bin/python" ]]; then
  echo "Existing pytorch environment not found: $CONDA_ENV_PATH" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Verified local model directory not found: $MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

finish() {
  local status=$?
  trap - EXIT
  echo "===== final nvidia-smi ====="
  nvidia-smi || true
  if [[ $status -eq 0 ]]; then
    echo "L20_TRANSFORMERS_SMOKE=success"
    echo "RESULT=$OUTPUT_DIR/result.json"
    echo "LOG=$LOG_FILE"
  else
    echo "L20_TRANSFORMERS_SMOKE=failed exit_code=$status" >&2
    echo "LOG=$LOG_FILE" >&2
  fi
  exit "$status"
}
trap finish EXIT

# Activate the audited existing environment. This script never installs or upgrades packages.
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
PYTHON_BIN="$(command -v python)"

echo "STARTED_AT_UTC=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MODEL_DIRECTORY=$(basename "$MODEL_PATH")"
echo "OUTPUT_DIR=$OUTPUT_DIR"

cd "$ROOT_DIR"
bash scripts/verify_l20_assets.sh \
  --model-dir "$MODEL_PATH" \
  --python-bin "$PYTHON_BIN"

"$PYTHON_BIN" scripts/test_l20_transformers.py \
  --model-path "$MODEL_PATH" \
  --fixture tests/fixtures/synthetic_patient.json \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --context-length 4096 \
  --max-new-tokens 256

echo "COMPLETED_AT_UTC=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

