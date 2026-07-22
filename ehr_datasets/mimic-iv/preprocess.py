from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mimiciv_preprocess_utils import (
    COLACARE_59_CATEGORICAL_LEVELS,
    preprocess_formatted_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert formatted MIMIC-IV episodes into ColaCare/pyehr pickles."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/mimic4_formatted_ehr.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/fold_1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--categorical-contract",
        choices=["observed", "colacare-59"],
        default="observed",
        help="Use training-observed categories or the recovered fixed 59-dimension contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_parquet.is_file():
        raise FileNotFoundError(f"Formatted parquet not found: {args.input_parquet}")
    frame = pd.read_parquet(args.input_parquet)
    levels = (
        COLACARE_59_CATEGORICAL_LEVELS
        if args.categorical_contract == "colacare-59"
        else None
    )
    result = preprocess_formatted_frame(
        frame,
        args.output_dir,
        seed=args.seed,
        categorical_levels=levels,
    )
    public_summary = {key: value for key, value in result.items() if key != "subject_splits"}
    print(json.dumps(public_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
