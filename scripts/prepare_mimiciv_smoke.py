from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mimiciv_preprocess_utils import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    REQUIRED_COLUMNS,
)


ITEM_TO_FEATURE = {
    223951: "Capillary refill rate",
    224308: "Capillary refill rate",
    220739: "Glascow coma scale eye opening",
    223901: "Glascow coma scale motor response",
    226755: "Glascow coma scale total",
    227013: "Glascow coma scale total",
    223900: "Glascow coma scale verbal response",
    228112: "Glascow coma scale verbal response",
    220051: "Diastolic blood pressure",
    220180: "Diastolic blood pressure",
    225310: "Diastolic blood pressure",
    224643: "Diastolic blood pressure",
    227242: "Diastolic blood pressure",
    223835: "Fraction inspired oxygen",
    220621: "Glucose",
    225664: "Glucose",
    226537: "Glucose",
    220045: "Heart Rate",
    226707: "Height",
    226730: "Height",
    220052: "Mean blood pressure",
    220181: "Mean blood pressure",
    220277: "Oxygen saturation",
    220210: "Respiratory rate",
    224688: "Respiratory rate",
    224689: "Respiratory rate",
    224690: "Respiratory rate",
    220050: "Systolic blood pressure",
    220179: "Systolic blood pressure",
    225309: "Systolic blood pressure",
    224167: "Systolic blood pressure",
    227243: "Systolic blood pressure",
    223761: "Temperature",
    223762: "Temperature",
    224639: "Weight",
    226512: "Weight",
    226531: "Weight",
    220274: "pH",
    223830: "pH",
}


PLAUSIBLE_RANGES = {
    "Diastolic blood pressure": (10.0, 250.0),
    "Fraction inspired oxygen": (0.20, 1.0),
    "Glucose": (10.0, 2000.0),
    "Heart Rate": (10.0, 350.0),
    "Height": (80.0, 250.0),
    "Mean blood pressure": (10.0, 300.0),
    "Oxygen saturation": (1.0, 100.0),
    "Respiratory rate": (1.0, 100.0),
    "Systolic blood pressure": (20.0, 350.0),
    "Temperature": (25.0, 45.0),
    "Weight": (20.0, 400.0),
    "pH": (6.5, 8.0),
}


def normalize_chart_value(itemid: int, value: float | int | None) -> float | None:
    """Normalize units for the selected MIMIC-IV chart item and reject outliers."""
    feature = ITEM_TO_FEATURE.get(int(itemid))
    if feature not in NUMERICAL_FEATURES or value is None or pd.isna(value):
        return None
    numeric = float(value)
    if int(itemid) == 223761:  # Fahrenheit -> Celsius
        numeric = (numeric - 32.0) * 5.0 / 9.0
    elif int(itemid) == 226707:  # inch -> cm
        numeric *= 2.54
    elif int(itemid) == 226531:  # lb -> kg
        numeric *= 0.45359237
    elif feature == "Fraction inspired oxygen" and numeric > 1.5:
        numeric /= 100.0

    low, high = PLAUSIBLE_RANGES[feature]
    if not np.isfinite(numeric) or numeric < low or numeric > high:
        return None
    return round(numeric, 3)


def _clean_category(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = " ".join(str(value).strip().split())
    return cleaned or None


def derive_gcs_total(eye: Any, motor: Any, verbal: Any) -> str | None:
    """Derive a categorical GCS total when all three component scores exist."""
    keyword_scores = {
        "eye": {
            "spontaneously": 4,
            "to speech": 3,
            "to pain": 2,
            "none": 1,
            "no response": 1,
        },
        "motor": {
            "obeys commands": 6,
            "localizes pain": 5,
            "flex-withdraws": 4,
            "withdraws": 4,
            "abnormal flexion": 3,
            "abnormal extension": 2,
            "no response": 1,
            "none": 1,
        },
        "verbal": {
            "oriented": 5,
            "confused": 4,
            "inappropriate words": 3,
            "incomprehensible sounds": 2,
            "no response": 1,
            "ett": 1,
            "none": 1,
        },
    }
    scores: list[int] = []
    for component, value in zip(("eye", "motor", "verbal"), (eye, motor, verbal)):
        if value is None or pd.isna(value):
            return None
        text = " ".join(str(value).strip().lower().split())
        match = re.match(r"\s*(\d+)", text)
        if match is not None:
            scores.append(int(match.group(1)))
            continue
        score = next(
            (
                candidate
                for keyword, candidate in keyword_scores[component].items()
                if keyword in text
            ),
            None,
        )
        if score is None:
            return None
        scores.append(score)
    total = sum(scores)
    return str(total) if 3 <= total <= 15 else None


def aggregate_chart_chunk(
    chunk: pd.DataFrame,
    episodes: pd.DataFrame,
    observation_hours: int,
) -> pd.DataFrame:
    """Filter and aggregate one chartevents chunk without returning raw identifiers."""
    if chunk.empty:
        return pd.DataFrame(columns=["RecordID", "hour", "feature", "value"])
    selected = chunk[
        chunk["stay_id"].isin(episodes.index)
        & chunk["itemid"].isin(ITEM_TO_FEATURE)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=["RecordID", "hour", "feature", "value"])

    episode_columns = episodes[
        ["stay_id", "subject_id", "intime", "RecordID"]
    ].reset_index(drop=True)
    selected = selected.merge(
        episode_columns,
        on="stay_id",
        how="inner",
        suffixes=("_event", "_episode"),
        validate="many_to_one",
    )
    selected = selected[
        selected["subject_id_event"] == selected["subject_id_episode"]
    ].copy()
    selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
    selected["hour"] = (
        (selected["charttime"] - selected["intime"]).dt.total_seconds() // 3600
    )
    selected = selected[
        selected["hour"].notna()
        & (selected["hour"] >= 0)
        & (selected["hour"] < observation_hours)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=["RecordID", "hour", "feature", "value"])

    selected["hour"] = selected["hour"].astype(int)
    selected["feature"] = selected["itemid"].map(ITEM_TO_FEATURE)
    selected["normalized"] = [
        _clean_category(raw_value)
        if feature in CATEGORICAL_FEATURES
        else normalize_chart_value(itemid, numeric_value)
        for itemid, feature, raw_value, numeric_value in zip(
            selected["itemid"],
            selected["feature"],
            selected["value"],
            selected["valuenum"],
        )
    ]
    selected = selected[selected["normalized"].notna()]

    rows: list[dict[str, Any]] = []
    for (record_id, hour, feature), group in selected.groupby(
        ["RecordID", "hour", "feature"], sort=False
    ):
        if feature in CATEGORICAL_FEATURES:
            value = group["normalized"].iloc[-1]
        else:
            value = float(pd.to_numeric(group["normalized"]).median())
        rows.append(
            {
                "RecordID": str(record_id),
                "hour": int(hour),
                "feature": str(feature),
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def _load_cohort(raw_root: Path, candidate_count: int) -> pd.DataFrame:
    patients = pd.read_csv(
        raw_root / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    admissions = pd.read_csv(
        raw_root / "hosp" / "admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime"],
    ).sort_values(["subject_id", "admittime"])
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    gap = admissions["next_admittime"] - admissions["dischtime"]
    admissions["Readmission"] = ((gap >= pd.Timedelta(0)) & (gap <= pd.Timedelta(days=30))).astype(int)

    icustays = pd.read_csv(
        raw_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    ).sort_values(["hadm_id", "intime"])
    icustays = icustays.drop_duplicates("hadm_id", keep="first")

    cohort = (
        icustays.merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
        .merge(patients, on="subject_id", validate="many_to_one")
    )
    cohort["Age"] = (
        cohort["anchor_age"]
        + cohort["admittime"].dt.year
        - cohort["anchor_year"]
    ).clip(lower=0, upper=120)
    cohort = cohort[
        (cohort["Age"] >= 18)
        & (cohort["los"] >= 2.0)
        & cohort["intime"].notna()
        & cohort["outtime"].notna()
    ].copy()
    cohort = cohort.sort_values(["subject_id", "intime"]).drop_duplicates(
        "subject_id", keep="first"
    )
    cohort = cohort.head(candidate_count).reset_index(drop=True)
    if len(cohort) < candidate_count:
        raise RuntimeError(
            f"Only {len(cohort)} eligible adult ICU episodes were found; requested {candidate_count}"
        )

    cohort["RecordID"] = [f"E{index:06d}" for index in range(1, len(cohort) + 1)]
    cohort["PatientID"] = [f"P{index:06d}" for index in range(1, len(cohort) + 1)]
    cohort["Sex"] = cohort["gender"].map({"F": 0, "M": 1}).astype(float)
    cohort["Outcome"] = cohort["hospital_expire_flag"].astype(int)
    cohort["LOS"] = cohort["los"].astype(float)
    return cohort


def _validate_dictionary(raw_root: Path) -> None:
    items = pd.read_csv(raw_root / "icu" / "d_items.csv.gz", usecols=["itemid", "label"])
    missing = sorted(set(ITEM_TO_FEATURE) - set(items["itemid"].astype(int)))
    if missing:
        raise RuntimeError(f"Configured item IDs are absent from d_items: {missing}")


def _finalize_cohort(
    candidates: pd.DataFrame,
    events: pd.DataFrame,
    cohort_size: int,
    minimum_observed_hours: int,
) -> pd.DataFrame:
    observed = events.groupby("RecordID")["hour"].nunique()
    eligible_ids = set(observed[observed >= minimum_observed_hours].index)
    eligible = candidates[candidates["RecordID"].isin(eligible_ids)].copy()
    if len(eligible) < cohort_size:
        raise RuntimeError(
            f"Only {len(eligible)} candidates have at least {minimum_observed_hours} observed hours; "
            f"requested {cohort_size}"
        )

    natural_positive_fraction = float(eligible["Outcome"].mean())
    positive_target = max(3, int(round(cohort_size * natural_positive_fraction)))
    positive_target = min(positive_target, int((eligible["Outcome"] == 1).sum()))
    negative_target = cohort_size - positive_target
    positives = eligible[eligible["Outcome"] == 1].head(positive_target)
    negatives = eligible[eligible["Outcome"] == 0].head(negative_target)
    if len(negatives) < negative_target:
        raise RuntimeError("Not enough negative episodes to assemble the requested cohort")
    return pd.concat([positives, negatives]).sort_values("subject_id").reset_index(drop=True)


def prepare_smoke_dataset(
    raw_root: Path,
    output_path: Path,
    cohort_size: int = 600,
    observation_hours: int = 48,
    chunksize: int = 1_000_000,
    candidate_multiplier: float = 1.5,
    minimum_observed_hours: int = 6,
    allow_sorted_early_stop: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    _validate_dictionary(raw_root)
    candidate_count = max(cohort_size, int(round(cohort_size * candidate_multiplier)))
    candidates = _load_cohort(raw_root, candidate_count)
    indexed_candidates = candidates.set_index("stay_id", drop=False)
    candidate_stays = set(candidates["stay_id"].astype(int))
    max_candidate_subject = int(candidates["subject_id"].max())

    event_parts: list[pd.DataFrame] = []
    rows_scanned = 0
    previous_subject_max: int | None = None
    sorted_so_far = True
    early_stopped = False
    chartevents_path = raw_root / "icu" / "chartevents.csv.gz"
    reader = pd.read_csv(
        chartevents_path,
        usecols=["subject_id", "stay_id", "itemid", "charttime", "value", "valuenum"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        rows_scanned += len(chunk)
        current_min = int(chunk["subject_id"].min())
        current_max = int(chunk["subject_id"].max())
        if previous_subject_max is not None and current_min < previous_subject_max:
            sorted_so_far = False
        previous_subject_max = current_max

        relevant = chunk[
            chunk["stay_id"].isin(candidate_stays)
            & chunk["itemid"].isin(ITEM_TO_FEATURE)
        ]
        if not relevant.empty:
            part = aggregate_chart_chunk(
                relevant,
                indexed_candidates,
                observation_hours=observation_hours,
            )
            if not part.empty:
                event_parts.append(part)

        if (
            allow_sorted_early_stop
            and sorted_so_far
            and current_min > max_candidate_subject
        ):
            early_stopped = True
            break

    if not event_parts:
        raise RuntimeError("No selected chart events were found for the candidate cohort")
    events = pd.concat(event_parts, ignore_index=True)
    events = (
        events.groupby(["RecordID", "hour", "feature"], as_index=False, sort=False)["value"]
        .last()
    )

    cohort = _finalize_cohort(
        candidates,
        events,
        cohort_size=cohort_size,
        minimum_observed_hours=minimum_observed_hours,
    )
    events = events[events["RecordID"].isin(set(cohort["RecordID"]))]
    pivot = events.pivot(index=["RecordID", "hour"], columns="feature", values="value")

    output_rows: list[dict[str, Any]] = []
    relative_epoch = pd.Timestamp("2000-01-01 00:00:00")
    for episode in cohort.itertuples(index=False):
        for hour in range(observation_hours):
            row = {
                "RecordID": episode.RecordID,
                "PatientID": episode.PatientID,
                "RecordTime": relative_epoch + pd.Timedelta(hours=hour),
                "Outcome": int(episode.Outcome),
                "LOS": float(episode.LOS),
                "Readmission": int(episode.Readmission),
                "Sex": float(episode.Sex),
                "Age": float(episode.Age),
            }
            for feature in [*CATEGORICAL_FEATURES, *NUMERICAL_FEATURES]:
                try:
                    row[feature] = pivot.loc[(episode.RecordID, hour), feature]
                except KeyError:
                    row[feature] = np.nan
            if pd.isna(row["Glascow coma scale total"]):
                derived_total = derive_gcs_total(
                    row["Glascow coma scale eye opening"],
                    row["Glascow coma scale motor response"],
                    row["Glascow coma scale verbal response"],
                )
                if derived_total is not None:
                    row["Glascow coma scale total"] = derived_total
            output_rows.append(row)

    formatted = pd.DataFrame(output_rows)[REQUIRED_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_parquet(output_path, index=False, compression="snappy")

    episode_labels = formatted.groupby("RecordID")["Outcome"].first()
    summary = {
        "dataset": "MIMIC-IV 2.2 local smoke cohort",
        "provisional_contract": True,
        "episode_definition": "first ICU stay per hospital admission; at most one episode per subject",
        "adult_minimum_age": 18,
        "minimum_icu_los_days": 2.0,
        "observation_hours": observation_hours,
        "time_bin_hours": 1,
        "cohort_episodes": int(formatted["RecordID"].nunique()),
        "cohort_subjects": int(formatted["PatientID"].nunique()),
        "formatted_rows": int(len(formatted)),
        "positive_outcomes": int(episode_labels.sum()),
        "negative_outcomes": int((episode_labels == 0).sum()),
        "chartevents_rows_scanned": int(rows_scanned),
        "chartevents_early_stopped": early_stopped,
        "chartevents_sorted_so_far": sorted_so_far,
        "raw_identifiers_written": False,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = output_path.with_name("formatted_ehr_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pseudonymized, provisional MIMIC-IV ICU smoke cohort."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("mimic-iv-2.2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/mimic4_formatted_ehr.parquet"),
    )
    parser.add_argument("--cohort-size", type=int, default=600)
    parser.add_argument("--observation-hours", type=int, default=48)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--minimum-observed-hours", type=int, default=6)
    parser.add_argument(
        "--full-chartevents-scan",
        action="store_true",
        help="Disable the smoke-only sorted subject early-stop optimization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_smoke_dataset(
        raw_root=args.raw_root,
        output_path=args.output,
        cohort_size=args.cohort_size,
        observation_hours=args.observation_hours,
        chunksize=args.chunksize,
        candidate_multiplier=args.candidate_multiplier,
        minimum_observed_hours=args.minimum_observed_hours,
        allow_sorted_early_stop=not args.full_chartevents_scan,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
