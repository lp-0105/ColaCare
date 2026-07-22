from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    "Capillary refill rate",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
]

NUMERICAL_FEATURES = [
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]

# Recovered from ColaCare commit 8548787^ (the original MIMIC-IV
# preprocess.py).  The checked-in hparams use lab_dim=59: these 47 historical
# one-hot columns plus the 12 numerical features above.  The strings are kept
# verbatim because model input order is part of the data contract.
COLACARE_59_CATEGORICAL_LEVELS = {
    "Capillary refill rate": ["0.0", "1.0"],
    "Glascow coma scale eye opening": [
        "To Pain",
        "3 To speech",
        "1 No Response",
        "4 Spontaneously",
        "None",
        "To Speech",
        "Spontaneously",
        "2 To pain",
    ],
    "Glascow coma scale motor response": [
        "1 No Response",
        "3 Abnorm flexion",
        "Abnormal extension",
        "No response",
        "4 Flex-withdraws",
        "Localizes Pain",
        "Flex-withdraws",
        "Obeys Commands",
        "Abnormal Flexion",
        "6 Obeys Commands",
        "5 Localizes Pain",
        "2 Abnorm extensn",
    ],
    "Glascow coma scale total": [
        "11",
        "10",
        "13",
        "12",
        "15",
        "14",
        "3",
        "5",
        "4",
        "7",
        "6",
        "9",
        "8",
    ],
    "Glascow coma scale verbal response": [
        "1 No Response",
        "No Response",
        "Confused",
        "Inappropriate Words",
        "Oriented",
        "No Response-ETT",
        "5 Oriented",
        "Incomprehensible sounds",
        "1.0 ET/Trach",
        "4 Confused",
        "2 Incomp sounds",
        "3 Inapprop words",
    ],
}


def formal_dynamic_columns() -> list[str]:
    """Return the exact 59-column dynamic input contract used by hparams.py."""
    return [
        f"{feature}->{level}"
        for feature in CATEGORICAL_FEATURES
        for level in COLACARE_59_CATEGORICAL_LEVELS[feature]
    ] + list(NUMERICAL_FEATURES)

REQUIRED_COLUMNS = [
    "RecordID",
    "PatientID",
    "RecordTime",
    "Outcome",
    "LOS",
    "Readmission",
    "Sex",
    "Age",
    *CATEGORICAL_FEATURES,
    *NUMERICAL_FEATURES,
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def split_subjects(
    frame: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
) -> dict[str, list[str]]:
    """Create deterministic, outcome-stratified subject-level splits."""
    test_fraction = 1.0 - train_fraction - val_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("train/val/test fractions must all be positive")

    subjects = (
        frame.groupby("PatientID", sort=True)["Outcome"]
        .max()
        .astype(int)
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for _, label_group in subjects.groupby("Outcome", sort=True):
        ids = label_group["PatientID"].astype(str).to_numpy(copy=True)
        rng.shuffle(ids)
        count = len(ids)
        if count < 3:
            raise ValueError(
                f"Outcome class has only {count} subjects; at least 3 are required"
            )

        val_count = max(1, int(round(count * val_fraction)))
        test_count = max(1, int(round(count * test_fraction)))
        if val_count + test_count >= count:
            val_count = 1
            test_count = 1
        train_count = count - val_count - test_count

        splits["train"].extend(ids[:train_count].tolist())
        splits["val"].extend(ids[train_count : train_count + val_count].tolist())
        splits["test"].extend(ids[train_count + val_count :].tolist())

    for name in splits:
        splits[name] = sorted(splits[name])

    sets = {name: set(values) for name, values in splits.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("subject leakage detected across splits")
    return splits


def _training_statistics(train_frame: pd.DataFrame) -> dict[str, Any]:
    normalize_features = ["Age", *NUMERICAL_FEATURES]
    mean = train_frame[normalize_features].apply(pd.to_numeric, errors="coerce").mean()
    std = train_frame[normalize_features].apply(pd.to_numeric, errors="coerce").std()
    median = train_frame[normalize_features].apply(pd.to_numeric, errors="coerce").median()
    std = std.mask(std.abs() < 1e-12, 1.0).fillna(1.0)
    mean = mean.fillna(0.0)
    median = median.fillna(mean)

    episode_los = (
        train_frame.groupby("RecordID", sort=False)["LOS"].first().astype(float)
    )
    los_mean = float(episode_los.mean())
    los_std = float(episode_los.std())
    if not np.isfinite(los_std) or los_std < 1e-12:
        los_std = 1.0

    return {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "median": median.to_dict(),
        "los_mean": los_mean,
        "los_std": los_std,
        "los_median": float(episode_los.median()),
        "large_los": float(episode_los.quantile(0.95)),
        "threshold": float(episode_los.mean() * 0.5),
        "source": "training subjects only",
    }


def _categorical_levels(train_frame: pd.DataFrame) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = train_frame[feature].dropna().astype(str).str.strip()
        levels[feature] = sorted(value for value in values.unique() if value)
        if not levels[feature]:
            levels[feature] = ["UNKNOWN_OBSERVED_CATEGORY"]
    return levels


def _encode_split(
    split_frame: pd.DataFrame,
    levels: dict[str, list[str]],
    stats: dict[str, Any],
) -> dict[str, list[Any]]:
    dynamic_columns = [
        f"{feature}->{level}"
        for feature in CATEGORICAL_FEATURES
        for level in levels[feature]
    ] + NUMERICAL_FEATURES
    model_columns = ["Sex", "Age", *dynamic_columns]

    all_x: list[np.ndarray] = []
    all_raw_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_pid: list[str] = []
    all_mask: list[np.ndarray] = []
    all_record_time: list[list[str]] = []

    for record_id, group in split_frame.groupby("RecordID", sort=True):
        group = group.sort_values("RecordTime").reset_index(drop=True)
        raw = pd.DataFrame(index=group.index)
        mask = pd.DataFrame(index=group.index)
        raw["Sex"] = pd.to_numeric(group["Sex"], errors="coerce")
        raw["Age"] = pd.to_numeric(group["Age"], errors="coerce")
        mask["Sex"] = raw["Sex"].isna().astype(float)
        mask["Age"] = raw["Age"].isna().astype(float)

        for feature in CATEGORICAL_FEATURES:
            source = group[feature]
            missing = source.isna()
            source_text = source.astype(str).str.strip()
            for level in levels[feature]:
                column = f"{feature}->{level}"
                raw[column] = (source_text == level).astype(float)
                mask[column] = missing.astype(float)

        for feature in NUMERICAL_FEATURES:
            raw[feature] = pd.to_numeric(group[feature], errors="coerce")
            mask[feature] = raw[feature].isna().astype(float)

        model = raw.copy()
        model["Sex"] = model["Sex"].fillna(float(stats["median"].get("Sex", 0.0)))
        model["Age"] = model["Age"].fillna(float(stats["median"]["Age"]))
        model["Age"] = (
            model["Age"] - float(stats["mean"]["Age"])
        ) / float(stats["std"]["Age"])

        for feature in NUMERICAL_FEATURES:
            model[feature] = model[feature].ffill().fillna(float(stats["median"][feature]))
            model[feature] = (
                model[feature] - float(stats["mean"][feature])
            ) / float(stats["std"][feature])

        outcome = int(group["Outcome"].iloc[0])
        readmission = int(group["Readmission"].iloc[0])
        los = float(group["LOS"].iloc[0])
        normalized_los = (los - float(stats["los_mean"])) / float(stats["los_std"])
        target = np.tile(
            np.array([outcome, normalized_los, readmission], dtype=np.float32),
            (len(group), 1),
        )

        all_x.append(model[model_columns].to_numpy(dtype=np.float32))
        all_raw_x.append(raw[model_columns].to_numpy(dtype=np.float32))
        all_y.append(target)
        all_pid.append(str(record_id))
        all_mask.append(mask[model_columns].to_numpy(dtype=np.float32))
        all_record_time.append(
            pd.to_datetime(group["RecordTime"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        )

    return {
        "x": all_x,
        "raw_x": all_raw_x,
        "y": all_y,
        "pid": all_pid,
        "missing_mask": all_mask,
        "record_time": all_record_time,
        "model_columns": model_columns,
        "dynamic_columns": dynamic_columns,
    }


def preprocess_formatted_frame(
    frame: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
    categorical_levels: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Convert formatted admission episodes into ColaCare/pyehr pickle files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"formatted parquet is missing columns: {missing_columns}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["RecordID"] = frame["RecordID"].astype(str)
    frame["PatientID"] = frame["PatientID"].astype(str)
    frame["RecordTime"] = pd.to_datetime(frame["RecordTime"], errors="raise")
    frame = frame.sort_values(["RecordID", "RecordTime"]).reset_index(drop=True)

    splits = split_subjects(frame, seed=seed)
    split_frames = {
        name: frame[frame["PatientID"].isin(subject_ids)].copy()
        for name, subject_ids in splits.items()
    }
    stats = _training_statistics(split_frames["train"])
    if categorical_levels is None:
        levels = _categorical_levels(split_frames["train"])
        categorical_contract = "training-observed"
    else:
        if set(categorical_levels) != set(CATEGORICAL_FEATURES):
            raise ValueError("categorical_levels must define exactly the five categorical features")
        levels = {
            feature: [str(level) for level in categorical_levels[feature]]
            for feature in CATEGORICAL_FEATURES
        }
        if any(not values or len(values) != len(set(values)) for values in levels.values()):
            raise ValueError("categorical_levels must contain non-empty unique values")
        categorical_contract = "fixed"

    unseen_categories = {
        feature: sorted(
            set(frame[feature].dropna().astype(str).str.strip()) - set(levels[feature])
        )
        for feature in CATEGORICAL_FEATURES
    }
    unseen_categories = {
        feature: values for feature, values in unseen_categories.items() if values
    }
    if unseen_categories:
        raise ValueError(f"formatted data contains categories outside fixed contract: {unseen_categories}")

    encoded: dict[str, dict[str, list[Any]]] = {}
    for name, split_frame in split_frames.items():
        encoded[name] = _encode_split(split_frame, levels, stats)
        for key in ["x", "raw_x", "y", "pid", "missing_mask", "record_time"]:
            pd.to_pickle(encoded[name][key], output_dir / f"{name}_{key}.pkl")

    dynamic_columns = encoded["train"]["dynamic_columns"]
    pd.to_pickle(NUMERICAL_FEATURES, output_dir / "numerical_features.pkl")
    pd.to_pickle(dynamic_columns, output_dir / "labtest_features.pkl")
    pd.to_pickle(levels, output_dir / "categorical_levels.pkl")
    pd.to_pickle(stats, output_dir / "normalization_stats.pkl")
    pd.to_pickle(
        {
            "los_mean": stats["los_mean"],
            "los_std": stats["los_std"],
            "los_median": stats["los_median"],
            "large_los": stats["large_los"],
            "threshold": stats["threshold"],
        },
        output_dir / "los_info.pkl",
    )
    pd.to_pickle(splits, output_dir / "subject_splits.pkl")

    basic = (
        frame[["RecordID", "Sex", "Age"]]
        .groupby("RecordID", sort=True)
        .first()
        .to_dict("index")
    )
    pd.to_pickle(basic, output_dir / "basic.pkl")

    numeric_stats = frame[["Outcome", *NUMERICAL_FEATURES]].copy()
    pd.to_pickle(
        numeric_stats[numeric_stats["Outcome"] == 0].describe().to_dict("dict"),
        output_dir / "survival.pkl",
    )
    pd.to_pickle(
        numeric_stats[numeric_stats["Outcome"] == 1].describe().to_dict("dict"),
        output_dir / "dead.pkl",
    )

    split_summary: dict[str, Any] = {}
    for name, split_frame in split_frames.items():
        episodes = split_frame.groupby("RecordID", sort=False).first()
        split_summary[name] = {
            "subjects": int(split_frame["PatientID"].nunique()),
            "episodes": int(split_frame["RecordID"].nunique()),
            "rows": int(len(split_frame)),
            "positive": int(episodes["Outcome"].sum()),
            "negative": int((episodes["Outcome"] == 0).sum()),
            "x_shape_first": list(np.asarray(encoded[name]["x"][0]).shape),
            "y_shape_first": list(np.asarray(encoded[name]["y"][0]).shape),
            "mask_shape_first": list(np.asarray(encoded[name]["missing_mask"][0]).shape),
        }

    missing_rates = {
        feature: float(frame[feature].isna().mean())
        for feature in [*CATEGORICAL_FEATURES, *NUMERICAL_FEATURES]
    }
    summary = {
        "seed": seed,
        "schema": REQUIRED_COLUMNS,
        "input_dim": 2 + len(dynamic_columns),
        "demo_dim": 2,
        "lab_dim": len(dynamic_columns),
        "splits": split_summary,
        "subject_intersections": {
            "train_val": len(set(splits["train"]) & set(splits["val"])),
            "train_test": len(set(splits["train"]) & set(splits["test"])),
            "val_test": len(set(splits["val"]) & set(splits["test"])),
        },
        "feature_missing_rate": missing_rates,
        "normalization_source": "training subjects only",
        "categorical_contract": categorical_contract,
        "categorical_level_count": int(sum(len(values) for values in levels.values())),
        "unseen_categories": unseen_categories,
        "notes_generated": False,
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        **summary,
        "subject_splits": splits,
    }
