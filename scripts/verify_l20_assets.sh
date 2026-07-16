#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: verify_l20_assets.sh --model-dir DIR [--python-bin PATH] [--checksum-file PATH] [--min-free-gib 2]

Verify an extracted local Transformers model and the existing CUDA Python environment.
No network request or package installation is performed.
EOF
}

MODEL_DIR=""
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/pytorch/bin/python}"
CHECKSUM_FILE=""
MIN_FREE_GIB=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --python-bin) PYTHON_BIN="${2:-}"; shift 2 ;;
    --checksum-file) CHECKSUM_FILE="${2:-}"; shift 2 ;;
    --min-free-gib) MIN_FREE_GIB="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "--model-dir is required" >&2
  exit 2
fi
if [[ "$MODEL_DIR" == *"://"* ]]; then
  echo "Model must be a local directory, not a URL or repository ID" >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Local model directory not found: $MODEL_DIR" >&2
  exit 2
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd -P)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for command_name in sha256sum df du find nvidia-smi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if ! [[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]]; then
  echo "--min-free-gib must be a non-negative integer" >&2
  exit 2
fi

for required in config.json tokenizer_config.json MODEL_MANIFEST.json SHA256SUMS; do
  if [[ ! -s "$MODEL_DIR/$required" ]]; then
    echo "Critical model file is missing or empty: $required" >&2
    exit 2
  fi
done
if ! find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +0c -print -quit | grep -q .; then
  echo "No non-empty safetensors weight files found" >&2
  exit 2
fi

echo "Verifying model snapshot SHA256 values"
(cd "$MODEL_DIR" && sha256sum -c SHA256SUMS)

if [[ -n "$CHECKSUM_FILE" ]]; then
  if [[ ! -s "$CHECKSUM_FILE" ]]; then
    echo "External checksum file is missing: $CHECKSUM_FILE" >&2
    exit 2
  fi
  CHECKSUM_FILE="$(cd "$(dirname "$CHECKSUM_FILE")" && pwd -P)/$(basename "$CHECKSUM_FILE")"
  echo "Verifying external asset SHA256 values"
  (cd "$(dirname "$CHECKSUM_FILE")" && sha256sum -c "$(basename "$CHECKSUM_FILE")")
fi

MODEL_SIZE_KIB="$(du -sk "$MODEL_DIR" | awk '{print $1}')"
FREE_KIB="$(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}')"
REQUIRED_FREE_KIB=$((MIN_FREE_GIB * 1024 * 1024))
if [[ -z "$FREE_KIB" ]] || (( FREE_KIB < REQUIRED_FREE_KIB )); then
  echo "Insufficient free disk space: ${FREE_KIB:-unknown} KiB available" >&2
  exit 2
fi
echo "MODEL_SIZE_KIB=$MODEL_SIZE_KIB"
echo "DISK_FREE_KIB=$FREE_KIB"

echo "Checking NVIDIA GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap \
  --format=csv,noheader

echo "Checking Python, PyTorch, Transformers, CUDA, BF16, and local tokenizer"
"$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
import json
from pathlib import Path
import sys

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

model_dir = Path(sys.argv[1]).resolve()
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.cuda.device_count() < 1:
    raise SystemExit("no CUDA devices are visible")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("torch.cuda.is_bf16_supported() is false")
config = AutoConfig.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=False
)
if not tokenizer.chat_template:
    raise SystemExit("local tokenizer has no chat template")
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "bf16": True,
    "model_type": config.model_type,
    "model_path_local": str(model_dir),
}, indent=2))
PY

echo "L20_ASSET_VERIFICATION=success"

