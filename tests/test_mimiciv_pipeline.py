from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from scripts.prepare_mimiciv_smoke import (
    aggregate_chart_chunk,
    derive_gcs_total,
    normalize_chart_value,
)
from scripts.train_mimiciv_ehr_smoke import (
    PeakRSSMonitor,
    cuda_device_index,
    initialize_cuda_for_memory_stats,
    pad_collate,
)
import psutil
from utils.mimiciv_preprocess_utils import (
    COLACARE_59_CATEGORICAL_LEVELS,
    preprocess_formatted_frame,
)


def test_normalize_chart_value_converts_units_and_rejects_implausible_values():
    assert normalize_chart_value(223761, 98.6) == 37.0  # Fahrenheit -> Celsius
    assert normalize_chart_value(226707, 70.0) == 177.8  # inch -> cm
    assert normalize_chart_value(226531, 220.462) == 100.0  # lb -> kg
    assert normalize_chart_value(223835, 40.0) == 0.4  # percent -> fraction
    assert normalize_chart_value(220045, 500.0) is None  # implausible heart rate


def test_derive_gcs_total_from_component_categories():
    assert derive_gcs_total("4 Spontaneously", "6 Obeys Commands", "5 Oriented") == "15"
    assert derive_gcs_total("Spontaneously", "Obeys Commands", "Oriented") == "15"
    assert derive_gcs_total("3 To Speech", None, "4 Confused") is None


def test_aggregate_chart_chunk_uses_relative_hours_and_never_exposes_raw_ids():
    episodes = pd.DataFrame(
        {
            "stay_id": [9001],
            "subject_id": [7001],
            "intime": pd.to_datetime(["2200-01-01 00:00:00"]),
            "RecordID": ["E000001"],
        }
    ).set_index("stay_id", drop=False)
    chunk = pd.DataFrame(
        {
            "subject_id": [7001, 7001, 7001],
            "stay_id": [9001, 9001, 9001],
            "itemid": [220045, 223762, 220045],
            "charttime": pd.to_datetime(
                ["2200-01-01 00:10:00", "2200-01-01 01:20:00", "2200-01-03 01:00:00"]
            ),
            "value": ["80", "37.2", "77"],
            "valuenum": [80.0, 37.2, 77.0],
        }
    )

    aggregated = aggregate_chart_chunk(chunk, episodes, observation_hours=48)

    assert set(aggregated["RecordID"]) == {"E000001"}
    assert set(aggregated["hour"]) == {0, 1}
    assert "subject_id" not in aggregated.columns
    assert "stay_id" not in aggregated.columns


def _synthetic_formatted_frame(subject_count: int = 40) -> pd.DataFrame:
    rows = []
    for subject_index in range(subject_count):
        outcome = int(subject_index % 5 == 0)
        for hour in range(3):
            rows.append(
                {
                    "RecordID": f"E{subject_index:06d}",
                    "PatientID": f"P{subject_index:06d}",
                    "RecordTime": pd.Timestamp("2000-01-01") + pd.Timedelta(hours=hour),
                    "Outcome": outcome,
                    "LOS": 3.0 + subject_index / 10,
                    "Readmission": 0,
                    "Sex": subject_index % 2,
                    "Age": 40.0 + subject_index,
                    "Capillary refill rate": "Normal <3 secs" if hour else np.nan,
                    "Glascow coma scale eye opening": "4 Spontaneously",
                    "Glascow coma scale motor response": "6 Obeys Commands",
                    "Glascow coma scale total": "15",
                    "Glascow coma scale verbal response": "5 Oriented",
                    "Diastolic blood pressure": 60.0 + hour,
                    "Fraction inspired oxygen": 0.21,
                    "Glucose": 100.0,
                    "Heart Rate": 70.0 + hour,
                    "Height": 170.0,
                    "Mean blood pressure": 80.0,
                    "Oxygen saturation": 98.0,
                    "Respiratory rate": 16.0,
                    "Systolic blood pressure": 120.0,
                    "Temperature": 37.0,
                    "Weight": 70.0,
                    "pH": 7.4,
                }
            )
    return pd.DataFrame(rows)


def test_preprocess_writes_compatible_pickles_without_subject_leakage(tmp_path: Path):
    frame = _synthetic_formatted_frame()
    result = preprocess_formatted_frame(frame, tmp_path, seed=42)

    required = [
        "train_x.pkl",
        "train_y.pkl",
        "train_pid.pkl",
        "val_x.pkl",
        "val_y.pkl",
        "val_pid.pkl",
        "test_x.pkl",
        "test_y.pkl",
        "test_pid.pkl",
        "train_missing_mask.pkl",
        "val_missing_mask.pkl",
        "test_missing_mask.pkl",
        "numerical_features.pkl",
        "labtest_features.pkl",
        "normalization_stats.pkl",
    ]
    assert all((tmp_path / name).is_file() for name in required)

    train_subjects = set(result["subject_splits"]["train"])
    val_subjects = set(result["subject_splits"]["val"])
    test_subjects = set(result["subject_splits"]["test"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)

    train_x = pd.read_pickle(tmp_path / "train_x.pkl")
    train_y = pd.read_pickle(tmp_path / "train_y.pkl")
    train_mask = pd.read_pickle(tmp_path / "train_missing_mask.pkl")
    assert len(train_x) == len(train_y) == len(train_mask)
    assert np.asarray(train_x[0]).shape[1] == result["input_dim"]
    assert np.asarray(train_y[0]).shape[1] == 3
    assert np.asarray(train_mask[0]).shape == np.asarray(train_x[0]).shape


def test_normalization_statistics_are_derived_only_from_training_subjects(tmp_path: Path):
    frame = _synthetic_formatted_frame(subject_count=60)
    baseline = preprocess_formatted_frame(frame, tmp_path / "baseline", seed=7)
    test_subjects = set(baseline["subject_splits"]["test"])

    modified = frame.copy()
    modified.loc[modified["PatientID"].isin(test_subjects), "Heart Rate"] = 9999.0
    changed = preprocess_formatted_frame(modified, tmp_path / "changed", seed=7)

    baseline_stats = pd.read_pickle(tmp_path / "baseline" / "normalization_stats.pkl")
    changed_stats = pd.read_pickle(tmp_path / "changed" / "normalization_stats.pkl")
    assert baseline["subject_splits"] == changed["subject_splits"]
    assert baseline_stats["mean"]["Heart Rate"] == changed_stats["mean"]["Heart Rate"]
    assert baseline_stats["std"]["Heart Rate"] == changed_stats["std"]["Heart Rate"]


def test_fixed_none_category_does_not_encode_missing_gcs_values(tmp_path: Path):
    frame = _synthetic_formatted_frame()
    frame["Capillary refill rate"] = "0.0"
    frame.loc[frame["RecordTime"].dt.hour == 0, "Glascow coma scale eye opening"] = "None"
    frame.loc[frame["RecordTime"].dt.hour == 1, "Glascow coma scale eye opening"] = None
    frame.loc[
        frame["RecordTime"].dt.hour == 2, "Glascow coma scale eye opening"
    ] = "Spontaneously"

    preprocess_formatted_frame(
        frame,
        tmp_path,
        seed=42,
        categorical_levels=COLACARE_59_CATEGORICAL_LEVELS,
    )
    dynamic = pd.read_pickle(tmp_path / "labtest_features.pkl")
    column = 2 + dynamic.index("Glascow coma scale eye opening->None")

    for split in ("train", "val", "test"):
        x = np.asarray(pd.read_pickle(tmp_path / f"{split}_x.pkl"), dtype=np.float32)
        mask = np.asarray(
            pd.read_pickle(tmp_path / f"{split}_missing_mask.pkl"), dtype=np.float32
        )
        assert np.array_equal(x[:, :, column], np.array([[1.0, 0.0, 0.0]] * len(x)))
        assert np.array_equal(
            mask[:, :, column], np.array([[0.0, 1.0, 0.0]] * len(mask))
        )


def test_smoke_collate_pads_sequences_and_preserves_last_outcome():
    batch = [
        (np.ones((2, 4), dtype=np.float32), np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)),
        (np.ones((1, 4), dtype=np.float32), np.array([[0, 0, 0]], dtype=np.float32)),
    ]
    x, y, lengths = pad_collate(batch)
    assert tuple(x.shape) == (2, 2, 4)
    assert tuple(y.shape) == (2, 2, 3)
    assert lengths.tolist() == [2, 1]
    assert y[0, lengths[0] - 1, 0].item() == 1.0


def test_cuda_device_index_is_an_integer_for_memory_apis():
    assert cuda_device_index(torch.device("cuda:0")) == 0


def test_cuda_memory_stats_initialization_selects_the_device():
    if not torch.cuda.is_available():
        return
    index = initialize_cuda_for_memory_stats(torch.device("cuda:0"))
    assert index == 0
    assert torch.cuda.is_initialized()


def test_peak_rss_monitor_records_at_least_current_process_memory():
    baseline = psutil.Process().memory_info().rss
    with PeakRSSMonitor() as monitor:
        payload = bytearray(1024 * 1024)
        assert len(payload) == 1024 * 1024
    assert monitor.peak_rss_bytes >= baseline


def test_direct_cli_entrypoints_can_resolve_repository_modules():
    repo_root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, str(repo_root / "scripts" / "prepare_mimiciv_smoke.py"), "--help"],
        [
            sys.executable,
            str(repo_root / "ehr_datasets" / "mimic-iv" / "preprocess.py"),
            "--help",
        ],
    ]
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
