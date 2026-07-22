from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mimiciv_preprocess_utils import (
    CATEGORICAL_FEATURES,
    COLACARE_59_CATEGORICAL_LEVELS,
)


def validate_processed(parquet_path: Path, data_dir: Path) -> dict[str, Any]:
    parquet = pd.read_parquet(
        parquet_path,
        columns=["RecordID", "PatientID", "Outcome", *CATEGORICAL_FEATURES],
    )
    episode_rows = parquet.groupby("RecordID", sort=False).size()
    episode_labels = parquet.groupby("RecordID", sort=False)["Outcome"].first().astype(int)
    record_to_subject = parquet.groupby("RecordID", sort=False)["PatientID"].first()
    if not (episode_rows == 48).all():
        raise AssertionError("not every formal episode has exactly 48 time bins")
    if parquet.groupby("RecordID")["Outcome"].nunique().max() != 1:
        raise AssertionError("Outcome changes within an episode")

    splits: dict[str, Any] = {}
    record_sets: dict[str, set[str]] = {}
    for mode in ["train", "val", "test"]:
        x = pd.read_pickle(data_dir / f"{mode}_x.pkl")
        y = pd.read_pickle(data_dir / f"{mode}_y.pkl")
        mask = pd.read_pickle(data_dir / f"{mode}_missing_mask.pkl")
        pid = [str(value) for value in pd.read_pickle(data_dir / f"{mode}_pid.pkl")]
        if not (len(x) == len(y) == len(mask) == len(pid)):
            raise AssertionError(f"{mode} x/y/mask/pid lengths differ")
        if len(pid) != len(set(pid)):
            raise AssertionError(f"{mode} contains duplicate RecordID values")
        record_sets[mode] = set(pid)

        x_shapes = {tuple(np.asarray(value).shape) for value in x}
        y_shapes = {tuple(np.asarray(value).shape) for value in y}
        mask_shapes = {tuple(np.asarray(value).shape) for value in mask}
        if x_shapes != {(48, 61)} or y_shapes != {(48, 3)} or mask_shapes != {(48, 61)}:
            raise AssertionError(f"{mode} contains an unexpected tensor shape")
        if any(not np.isfinite(np.asarray(value)).all() for value in x):
            raise AssertionError(f"{mode}_x contains NaN or inf")
        if any(not np.isfinite(np.asarray(value)).all() for value in y):
            raise AssertionError(f"{mode}_y contains NaN or inf")
        if any(not np.isfinite(np.asarray(value)).all() for value in mask):
            raise AssertionError(f"{mode}_missing_mask contains NaN or inf")

        pickle_labels = np.array([int(np.asarray(value)[-1, 0]) for value in y])
        expected_labels = episode_labels.loc[pid].to_numpy(dtype=int)
        if not np.array_equal(pickle_labels, expected_labels):
            raise AssertionError(f"{mode} Outcome labels do not match formatted parquet")
        splits[mode] = {
            "episodes": len(pid),
            "positive": int(pickle_labels.sum()),
            "negative": int((pickle_labels == 0).sum()),
            "positive_rate": float(pickle_labels.mean()),
            "x_shape": [48, 61],
            "y_shape": [48, 3],
            "mask_shape": [48, 61],
            "finite": True,
            "record_ids_unique": True,
        }
        del x, y, mask

    intersections = {
        "train_val": len(record_sets["train"] & record_sets["val"]),
        "train_test": len(record_sets["train"] & record_sets["test"]),
        "val_test": len(record_sets["val"] & record_sets["test"]),
    }
    if any(intersections.values()):
        raise AssertionError("RecordID leakage detected across splits")

    subject_splits = pd.read_pickle(data_dir / "subject_splits.pkl")
    subject_sets = {name: set(map(str, values)) for name, values in subject_splits.items()}
    subject_intersections = {
        "train_val": len(subject_sets["train"] & subject_sets["val"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "val_test": len(subject_sets["val"] & subject_sets["test"]),
    }
    if any(subject_intersections.values()):
        raise AssertionError("PatientID leakage detected across splits")

    observed_levels = {
        feature: sorted(parquet[feature].dropna().astype(str).unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    inactive_levels = {
        feature: [
            value
            for value in COLACARE_59_CATEGORICAL_LEVELS[feature]
            if value not in set(observed_levels[feature])
        ]
        for feature in CATEGORICAL_FEATURES
    }
    inactive_levels = {key: value for key, value in inactive_levels.items() if value}
    report = {
        "status": "PASS",
        "formatted_rows": int(len(parquet)),
        "episodes": int(len(episode_rows)),
        "subjects": int(record_to_subject.nunique()),
        "all_episode_lengths_48": True,
        "outcome_constant_within_episode": True,
        "splits": splits,
        "record_intersections": intersections,
        "subject_intersections": subject_intersections,
        "lab_dim": 59,
        "input_dim": 61,
        "observed_categorical_level_count": int(sum(map(len, observed_levels.values()))),
        "fixed_categorical_level_count": 47,
        "inactive_fixed_level_count": int(sum(map(len, inactive_levels.values()))),
        "observed_categorical_levels": observed_levels,
        "inactive_fixed_levels": inactive_levels,
        "notes_generated": False,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal MIMIC-IV ColaCare pickles.")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_processed(args.parquet, args.data_dir)
    output = args.output or args.data_dir / "validation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
