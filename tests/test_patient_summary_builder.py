from __future__ import annotations

import numpy as np

from tests.adapter_fixtures import synthetic_case
from utils.patient_summary_builder import build_patient_summary


def test_raw_values_and_missing_mask_drive_deterministic_summary():
    case = synthetic_case()
    summary = build_patient_summary(
        case["raw_x"],
        case["missing_mask"],
        case["feature_names"],
        case["config"],
    )

    assert summary["age"] == {
        "age_band": "65-74",
        "observed_status": "observed",
    }
    assert summary["sex"] == {
        "value": "male",
        "observed_status": "observed",
    }
    heart_rate = summary["continuous_features"]["Heart Rate"]
    assert heart_rate["first_observed"] == 80.0
    assert heart_rate["latest_observed"] == 110.0
    assert heart_rate["min_observed"] == 80.0
    assert heart_rate["max_observed"] == 110.0
    assert heart_rate["observed_count"] == 3
    assert heart_rate["trend"] == {
        "method": "latest_minus_earliest",
        "delta": 30.0,
        "direction": "increasing",
    }
    assert summary["continuous_features"]["pH"]["observed_status"] == "missing"
    assert summary["continuous_features"]["pH"]["latest_observed"] is None

    eye = summary["categorical_features"]["Glascow coma scale eye opening"]
    assert eye["first_observed_state"] == "To Pain"
    assert eye["latest_observed_state"] == "Spontaneously"
    assert eye["observed_state_changes"] == 1
    assert eye["observed_count"] == 2


def test_empty_measurement_patient_is_supported_without_inventing_values():
    case = synthetic_case(empty=True)
    summary = build_patient_summary(
        case["raw_x"],
        case["missing_mask"],
        case["feature_names"],
        case["config"],
    )

    assert summary["age"]["observed_status"] == "missing"
    assert summary["sex"]["observed_status"] == "missing"
    assert all(
        item["observed_status"] == "missing"
        for item in summary["continuous_features"].values()
    )
    assert all(
        item["observed_status"] == "missing"
        for item in summary["categorical_features"].values()
    )


def test_shape_and_nonfinite_observed_values_are_rejected():
    case = synthetic_case()
    with np.testing.assert_raises_regex(ValueError, "shape"):
        build_patient_summary(
            case["raw_x"][:, :-1],
            case["missing_mask"],
            case["feature_names"],
            case["config"],
        )

    raw_x = case["raw_x"].copy()
    raw_x[0, case["feature_names"].index("Heart Rate")] = np.inf
    with np.testing.assert_raises_regex(ValueError, "non-finite observed"):
        build_patient_summary(
            raw_x,
            case["missing_mask"],
            case["feature_names"],
            case["config"],
        )
