#!/usr/bin/env bash
set -euo pipefail

: "${COLACARE_ROOT:?Set COLACARE_ROOT to the extracted ColaCare repository}"
: "${MIMIC_SMOKE_ROOT:?Set MIMIC_SMOKE_ROOT to the extracted mimiciv-smoke-600 directory}"

PYTORCH_ENV="${PYTORCH_ENV:-/opt/miniconda3/envs/pytorch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/colacare_mimiciv_smoke}"
VENDOR_ROOT="${VENDOR_ROOT:-$HOME/colacare_vendor/mimiciv}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENDOR_ROOT:$COLACARE_ROOT:${PYTHONPATH:-}"
export MIMIC_SMOKE_ROOT

mkdir -p "$OUTPUT_ROOT"
source /opt/miniconda3/bin/activate
conda activate "$PYTORCH_ENV"
cd "$COLACARE_ROOT"

python - <<'PY' | tee "$OUTPUT_ROOT/environment_report.txt"
import json
import platform
import sys
import numpy
import pandas
import psutil
import torch

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "psutil": psutil.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(report, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

python - <<'PY'
import os
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(os.environ["MIMIC_SMOKE_ROOT"]) / "fold_1"
for mode in ("train", "val", "test"):
    x = pd.read_pickle(root / f"{mode}_x.pkl")
    y = pd.read_pickle(root / f"{mode}_y.pkl")
    mask = pd.read_pickle(root / f"{mode}_missing_mask.pkl")
    assert len(x) == len(y) == len(mask)
    assert np.asarray(x[0]).shape == np.asarray(mask[0]).shape
    assert np.asarray(y[0]).shape[1] == 3
    assert all(np.isfinite(np.asarray(value)).all() for value in x)
    print(mode, "episodes=", len(x), "x_shape=", np.asarray(x[0]).shape)
PY

python scripts/train_mimiciv_ehr_smoke.py \
  --data-dir "$MIMIC_SMOKE_ROOT/fold_1" \
  --output-dir "$OUTPUT_ROOT/checkpoint" \
  --epochs 1 \
  --batch-size 16 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --seed 42 \
  2>&1 | tee "$OUTPUT_ROOT/retain_smoke.log"

nvidia-smi | tee "$OUTPUT_ROOT/nvidia-smi.txt"
echo "L20 MIMIC-IV 600-episode RETAIN smoke PASS"
