from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_mimiciv_smoke import (
    ITEM_TO_FEATURE,
    PLAUSIBLE_RANGES,
    derive_gcs_total,
    normalize_chart_value,
)
from utils.mimiciv_preprocess_utils import (
    CATEGORICAL_FEATURES,
    COLACARE_59_CATEGORICAL_LEVELS,
    NUMERICAL_FEATURES,
    REQUIRED_COLUMNS,
    formal_dynamic_columns,
)


LAB_ITEM_TO_FEATURE = {
    50809: "Glucose",       # blood gas, Blood
    50931: "Glucose",       # chemistry, Blood
    52027: "Glucose",       # whole blood, Blood
    52569: "Glucose",       # chemistry, Blood
    50820: "pH",            # blood gas, Blood
}

SOURCE_PRIORITY = {"chartevents": 0, "labevents": 1}


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def normalize_formal_category(feature: str, value: Any) -> str | None:
    """Map MIMIC-IV values into the exact historical ColaCare 47-level contract."""
    text = _clean_text(value)
    if text is None:
        return None

    if feature == "Capillary refill rate":
        lowered = text.lower()
        if "abnormal" in lowered or lowered in {"1", "1.0"}:
            text = "1.0"
        elif "normal" in lowered or lowered in {"0", "0.0"}:
            text = "0.0"
    elif feature == "Glascow coma scale eye opening":
        aliases = {
            "no response": "1 No Response",
            "1 no response": "1 No Response",
            "3 to speech": "3 To speech",
            "4 spontaneously": "4 Spontaneously",
            "2 to pain": "2 To pain",
            "to speech": "To Speech",
            "to pain": "To Pain",
            "spontaneously": "Spontaneously",
            "none": "None",
        }
        text = aliases.get(text.lower(), text)
    elif feature == "Glascow coma scale total":
        try:
            numeric = float(text)
            if numeric.is_integer():
                text = str(int(numeric))
        except ValueError:
            pass

    return text if text in COLACARE_59_CATEGORICAL_LEVELS[feature] else None


def should_replace_event(
    existing_priority: int,
    existing_charttime_ns: int,
    incoming_priority: int,
    incoming_charttime_ns: int,
) -> bool:
    """Resolve a feature/hour collision deterministically."""
    return incoming_priority > existing_priority or (
        incoming_priority == existing_priority
        and incoming_charttime_ns >= existing_charttime_ns
    )


def build_formal_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    adult_minimum_age: int = 18,
    minimum_icu_los_hours: int = 48,
) -> pd.DataFrame:
    """Select the first ICU stay per subject, then apply adult/LOS eligibility."""
    patients = patients.copy()
    admissions = admissions.copy()
    icustays = icustays.copy()
    for column in ["admittime", "dischtime"]:
        admissions[column] = pd.to_datetime(admissions[column], errors="coerce")
    for column in ["intime", "outtime"]:
        icustays[column] = pd.to_datetime(icustays[column], errors="coerce")

    admissions = admissions.sort_values(["subject_id", "admittime", "hadm_id"])
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    gap = admissions["next_admittime"] - admissions["dischtime"]
    admissions["Readmission"] = (
        (gap >= pd.Timedelta(0)) & (gap <= pd.Timedelta(days=30))
    ).astype(int)

    cohort = (
        icustays.merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
        .merge(patients, on="subject_id", validate="many_to_one")
        .sort_values(["subject_id", "intime", "stay_id"])
        .drop_duplicates("subject_id", keep="first")
    )
    cohort["Age"] = (
        cohort["anchor_age"]
        + cohort["admittime"].dt.year
        - cohort["anchor_year"]
    )
    cohort = cohort[
        (cohort["Age"] >= adult_minimum_age)
        & (cohort["Age"] <= 120)
        & (cohort["los"] * 24.0 >= minimum_icu_los_hours)
        & cohort["intime"].notna()
        & cohort["outtime"].notna()
    ].copy()
    cohort = cohort.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    cohort["episode_idx"] = np.arange(len(cohort), dtype=np.int64)
    cohort["RecordID"] = [f"E{index:06d}" for index in range(1, len(cohort) + 1)]
    cohort["PatientID"] = [f"P{index:06d}" for index in range(1, len(cohort) + 1)]
    cohort["Sex"] = cohort["gender"].map({"F": 0.0, "M": 1.0})
    cohort["Outcome"] = cohort["hospital_expire_flag"].astype(int)
    cohort["LOS"] = cohort["los"].astype(float)
    return cohort


def load_formal_cohort(raw_root: Path, config: dict[str, Any]) -> pd.DataFrame:
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
    )
    icustays = pd.read_csv(
        raw_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    )
    return build_formal_cohort(
        patients,
        admissions,
        icustays,
        adult_minimum_age=int(config["adult_minimum_age"]),
        minimum_icu_los_hours=int(config["minimum_icu_los_hours"]),
    )


def _rss_gib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def _log(message: str, log_handle) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    log_handle.write(line + "\n")
    log_handle.flush()


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            episode_idx INTEGER NOT NULL,
            hour INTEGER NOT NULL,
            feature TEXT NOT NULL,
            value TEXT NOT NULL,
            source_priority INTEGER NOT NULL,
            charttime_ns INTEGER NOT NULL,
            PRIMARY KEY (episode_idx, hour, feature)
        )
        """
    )
    return connection


UPSERT_SQL = """
INSERT INTO events (episode_idx, hour, feature, value, source_priority, charttime_ns)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(episode_idx, hour, feature) DO UPDATE SET
    value=excluded.value,
    source_priority=excluded.source_priority,
    charttime_ns=excluded.charttime_ns
WHERE excluded.source_priority > events.source_priority
   OR (excluded.source_priority = events.source_priority
       AND excluded.charttime_ns >= events.charttime_ns)
"""


def _normalize_lab_value(feature: str, value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    low, high = PLAUSIBLE_RANGES[feature]
    if not np.isfinite(numeric) or numeric < low or numeric > high:
        return None
    return round(numeric, 3)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _scan_chartevents(
    raw_root: Path,
    cohort: pd.DataFrame,
    connection: sqlite3.Connection,
    state: dict[str, Any],
    state_path: Path,
    chunksize: int,
    observation_hours: int,
    log_handle,
) -> None:
    source = "chartevents"
    if state.get(source, {}).get("complete"):
        _log("chartevents already complete; resume skips this source", log_handle)
        return
    stay_to_episode = dict(zip(cohort["stay_id"].astype(int), cohort["episode_idx"].astype(int)))
    stay_to_subject = dict(zip(cohort["stay_id"].astype(int), cohort["subject_id"].astype(int)))
    episode_intime = cohort.set_index("episode_idx")["intime"]
    scanned = 0
    hits = 0
    started = time.perf_counter()
    reader = pd.read_csv(
        raw_root / "icu" / "chartevents.csv.gz",
        usecols=["subject_id", "stay_id", "itemid", "charttime", "value", "valuenum"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        scanned += len(chunk)
        selected = chunk[
            chunk["stay_id"].isin(stay_to_episode)
            & chunk["itemid"].isin(ITEM_TO_FEATURE)
        ].copy()
        if not selected.empty:
            selected["episode_idx"] = selected["stay_id"].map(stay_to_episode)
            selected = selected[
                selected["subject_id"].astype(int)
                == selected["stay_id"].map(stay_to_subject).astype(int)
            ]
            selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
            selected["intime"] = selected["episode_idx"].map(episode_intime)
            selected["hour"] = (
                (selected["charttime"] - selected["intime"]).dt.total_seconds() // 3600
            )
            selected = selected[
                selected["hour"].notna()
                & (selected["hour"] >= 0)
                & (selected["hour"] < observation_hours)
            ]
            rows = []
            for item in selected.itertuples(index=False):
                feature = ITEM_TO_FEATURE[int(item.itemid)]
                if feature in CATEGORICAL_FEATURES:
                    value = normalize_formal_category(feature, item.value)
                else:
                    value = normalize_chart_value(int(item.itemid), item.valuenum)
                if value is None:
                    continue
                rows.append(
                    (
                        int(item.episode_idx),
                        int(item.hour),
                        feature,
                        str(value),
                        SOURCE_PRIORITY[source],
                        int(pd.Timestamp(item.charttime).value),
                    )
                )
            if rows:
                connection.executemany(UPSERT_SQL, rows)
                connection.commit()
                hits += len(rows)
        state[source] = {
            "complete": False,
            "chunks_scanned": chunk_number,
            "rows_scanned": scanned,
            "valid_events_seen": hits,
        }
        _write_state(state_path, state)
        if chunk_number == 1 or chunk_number % 5 == 0:
            _log(
                f"source={source} FULL_SCAN=true rows_scanned={scanned:,} "
                f"valid_events_seen={hits:,} rss_gib={_rss_gib():.3f} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                log_handle,
            )
    state[source]["complete"] = True
    state[source]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    _write_state(state_path, state)


def _scan_labevents(
    raw_root: Path,
    cohort: pd.DataFrame,
    connection: sqlite3.Connection,
    state: dict[str, Any],
    state_path: Path,
    chunksize: int,
    observation_hours: int,
    log_handle,
) -> None:
    source = "labevents"
    if state.get(source, {}).get("complete"):
        _log("labevents already complete; resume skips this source", log_handle)
        return
    hadm_to_episode = dict(zip(cohort["hadm_id"].astype(int), cohort["episode_idx"].astype(int)))
    episode_intime = cohort.set_index("episode_idx")["intime"]
    scanned = 0
    hits = 0
    started = time.perf_counter()
    reader = pd.read_csv(
        raw_root / "hosp" / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        scanned += len(chunk)
        selected = chunk[
            chunk["hadm_id"].isin(hadm_to_episode)
            & chunk["itemid"].isin(LAB_ITEM_TO_FEATURE)
        ].copy()
        if not selected.empty:
            selected["episode_idx"] = selected["hadm_id"].map(hadm_to_episode)
            selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
            selected["intime"] = selected["episode_idx"].map(episode_intime)
            selected["hour"] = (
                (selected["charttime"] - selected["intime"]).dt.total_seconds() // 3600
            )
            selected = selected[
                selected["hour"].notna()
                & (selected["hour"] >= 0)
                & (selected["hour"] < observation_hours)
            ]
            rows = []
            for item in selected.itertuples(index=False):
                feature = LAB_ITEM_TO_FEATURE[int(item.itemid)]
                value = _normalize_lab_value(feature, item.valuenum)
                if value is None:
                    continue
                rows.append(
                    (
                        int(item.episode_idx),
                        int(item.hour),
                        feature,
                        str(value),
                        SOURCE_PRIORITY[source],
                        int(pd.Timestamp(item.charttime).value),
                    )
                )
            if rows:
                connection.executemany(UPSERT_SQL, rows)
                connection.commit()
                hits += len(rows)
        state[source] = {
            "complete": False,
            "chunks_scanned": chunk_number,
            "rows_scanned": scanned,
            "valid_events_seen": hits,
        }
        _write_state(state_path, state)
        if chunk_number == 1 or chunk_number % 5 == 0:
            _log(
                f"source={source} FULL_SCAN=true rows_scanned={scanned:,} "
                f"valid_events_seen={hits:,} rss_gib={_rss_gib():.3f} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                log_handle,
            )
    state[source]["complete"] = True
    state[source]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    _write_state(state_path, state)


def _iter_episode_batches(cohort: pd.DataFrame, size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(cohort), size):
        yield cohort.iloc[start : start + size]


def _write_formatted_parquet(
    cohort: pd.DataFrame,
    connection: sqlite3.Connection,
    output_path: Path,
    observation_hours: int,
    log_handle,
    episode_batch_size: int = 500,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = None
    relative_epoch = pd.Timestamp("2000-01-01")
    try:
        for batch_number, batch in enumerate(_iter_episode_batches(cohort, episode_batch_size), start=1):
            first_index = int(batch["episode_idx"].min())
            last_index = int(batch["episode_idx"].max())
            fetched = connection.execute(
                "SELECT episode_idx, hour, feature, value FROM events "
                "WHERE episode_idx BETWEEN ? AND ?",
                (first_index, last_index),
            ).fetchall()
            event_map = {
                (int(episode_idx), int(hour), str(feature)): str(value)
                for episode_idx, hour, feature, value in fetched
            }
            rows: list[dict[str, Any]] = []
            for episode in batch.itertuples(index=False):
                for hour in range(observation_hours):
                    row: dict[str, Any] = {
                        "RecordID": episode.RecordID,
                        "PatientID": episode.PatientID,
                        "RecordTime": relative_epoch + pd.Timedelta(hours=hour),
                        "Outcome": int(episode.Outcome),
                        "LOS": float(episode.LOS),
                        "Readmission": int(episode.Readmission),
                        "Sex": float(episode.Sex),
                        "Age": float(episode.Age),
                    }
                    for feature in CATEGORICAL_FEATURES:
                        row[feature] = event_map.get((episode.episode_idx, hour, feature))
                    for feature in NUMERICAL_FEATURES:
                        value = event_map.get((episode.episode_idx, hour, feature))
                        row[feature] = float(value) if value is not None else np.nan
                    if row["Glascow coma scale total"] is None:
                        row["Glascow coma scale total"] = derive_gcs_total(
                            row["Glascow coma scale eye opening"],
                            row["Glascow coma scale motor response"],
                            row["Glascow coma scale verbal response"],
                        )
                    rows.append(row)
            frame = pd.DataFrame(rows)[REQUIRED_COLUMNS]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="snappy")
            writer.write_table(table)
            if batch_number == 1 or batch_number % 10 == 0:
                _log(
                    f"parquet_batches={batch_number} episodes_written={min(batch_number * episode_batch_size, len(cohort)):,} "
                    f"rss_gib={_rss_gib():.3f}",
                    log_handle,
                )
    finally:
        if writer is not None:
            writer.close()
    temporary_path.replace(output_path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _write_metadata(
    raw_root: Path,
    output_path: Path,
    cohort: pd.DataFrame,
    config: dict[str, Any],
    state: dict[str, Any],
    started: float,
) -> None:
    episode_labels = cohort["Outcome"].astype(int)
    summary = {
        "dataset": "MIMIC-IV 2.2",
        "mode": "formal",
        "FULL_SCAN": True,
        "sample_unit": config["sample_unit"],
        "cohort_episodes": int(len(cohort)),
        "cohort_subjects": int(len(cohort)),
        "positive_outcomes": int(episode_labels.sum()),
        "negative_outcomes": int((episode_labels == 0).sum()),
        "positive_rate": float(episode_labels.mean()),
        "formatted_rows": int(len(cohort) * int(config["observation_hours"])),
        "raw_identifiers_written": False,
        "scan": state,
        "output_path": str(output_path),
        "output_bytes": int(output_path.stat().st_size),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_path.parent / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    mapping = {
        "version": "colacare-mimiciv-formal-v1",
        "historical_contract_source": "git show 8548787^:ehr_datasets/mimic-iv/preprocess.py",
        "categorical_levels": COLACARE_59_CATEGORICAL_LEVELS,
        "categorical_columns": formal_dynamic_columns()[:-len(NUMERICAL_FEATURES)],
        "numerical_features": NUMERICAL_FEATURES,
        "chartevents_item_to_feature": {str(k): v for k, v in ITEM_TO_FEATURE.items()},
        "labevents_item_to_feature": {str(k): v for k, v in LAB_ITEM_TO_FEATURE.items()},
        "labevents_policy": "Blood glucose and blood pH only; labevents wins within the same feature/hour",
        "plausible_ranges": PLAUSIBLE_RANGES,
        "lab_dim": 59,
    }
    (output_path.parent / "feature_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    inputs = [
        "hosp/patients.csv.gz",
        "hosp/admissions.csv.gz",
        "icu/icustays.csv.gz",
        "icu/chartevents.csv.gz",
        "icu/d_items.csv.gz",
        "hosp/labevents.csv.gz",
        "hosp/d_labitems.csv.gz",
    ]
    manifest = {
        "git_commit": _git_commit(),
        "raw_data_version": "MIMIC-IV 2.2",
        "raw_sha256_conclusion": "33/33 PASS in prior read-only full audit; hashes not recomputed by this generation run",
        "raw_sha256_manifest": str(raw_root / "SHA256SUMS.txt"),
        "cohort_parameters": config,
        "feature_mapping_version": "colacare-mimiciv-formal-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": [
            output_path.name,
            "cohort_summary.json",
            "feature_mapping.json",
            "preprocessing_manifest.json",
        ],
        "random_seed": int(config["random_seed"]),
        "FULL_SCAN": True,
    }
    (output_path.parent / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def prepare_formal_dataset(
    raw_root: Path,
    output_path: Path,
    config_path: Path,
    chunksize: int = 1_000_000,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("full_scan") is not True or config.get("cohort_limit") is not None:
        raise ValueError("formal config must set full_scan=true and cohort_limit=null")
    if len(formal_dynamic_columns()) != int(config["lab_dim"]):
        raise ValueError("formal feature contract does not match configured lab_dim")

    work_dir = output_path.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_path.parent / "formal_raw_to_parquet.log"
    state_path = work_dir / "scan_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    with log_path.open("a", encoding="utf-8") as log_handle:
        _log("FULL_SCAN=true", log_handle)
        _log(f"config={config_path} raw_root={raw_root} output={output_path}", log_handle)
        cohort = load_formal_cohort(raw_root, config)
        _log(
            f"cohort_episodes={len(cohort):,} positives={int(cohort['Outcome'].sum()):,} "
            f"positive_rate={cohort['Outcome'].mean():.6f} rss_gib={_rss_gib():.3f}",
            log_handle,
        )
        connection = _open_db(work_dir / "formal_events.sqlite3")
        try:
            _scan_chartevents(
                raw_root,
                cohort,
                connection,
                state,
                state_path,
                chunksize,
                int(config["observation_hours"]),
                log_handle,
            )
            if config.get("include_labevents"):
                _scan_labevents(
                    raw_root,
                    cohort,
                    connection,
                    state,
                    state_path,
                    chunksize,
                    int(config["observation_hours"]),
                    log_handle,
                )
            _write_formatted_parquet(
                cohort,
                connection,
                output_path,
                int(config["observation_hours"]),
                log_handle,
            )
        finally:
            connection.close()
        _write_metadata(raw_root, output_path, cohort, config, state, started)
        _log(
            f"formal raw-to-parquet complete bytes={output_path.stat().st_size:,} "
            f"elapsed_seconds={time.perf_counter() - started:.1f}",
            log_handle,
        )
    return json.loads((output_path.parent / "cohort_summary.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-scan formal MIMIC-IV raw-to-parquet preparation.")
    parser.add_argument("--raw-root", type=Path, default=Path("mimic-iv-2.2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/formal/mimic4_formatted_ehr.parquet"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mimiciv_formal.json"),
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_formal_dataset(args.raw_root, args.output, args.config, args.chunksize)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
