from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.prepare_mimiciv_formal import (
    build_formal_cohort,
    normalize_formal_category,
    should_replace_event,
)
from utils.mimiciv_preprocess_utils import (
    COLACARE_59_CATEGORICAL_LEVELS,
    NUMERICAL_FEATURES,
    formal_dynamic_columns,
)


def test_historical_colacare_contract_is_47_categories_plus_12_numeric():
    category_count = sum(len(values) for values in COLACARE_59_CATEGORICAL_LEVELS.values())
    columns = formal_dynamic_columns()

    assert category_count == 47
    assert len(NUMERICAL_FEATURES) == 12
    assert len(columns) == 59
    assert columns[:2] == [
        "Capillary refill rate->0.0",
        "Capillary refill rate->1.0",
    ]
    assert columns[-1] == "pH"


def test_formal_category_mapping_uses_only_recovered_historical_levels():
    assert normalize_formal_category("Capillary refill rate", "Normal <3 Seconds") == "0.0"
    assert normalize_formal_category("Capillary refill rate", "Abnormal >3 Seconds") == "1.0"
    assert normalize_formal_category("Glascow coma scale eye opening", "No Response") == "1 No Response"
    assert normalize_formal_category("Glascow coma scale motor response", "6 Obeys Commands") == "6 Obeys Commands"
    assert normalize_formal_category("Glascow coma scale total", "15.0") == "15"
    assert normalize_formal_category("Glascow coma scale verbal response", "unexpected") is None


def test_formal_cohort_selects_first_icu_per_subject_before_los_filter():
    patients = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "gender": ["F", "M", "F"],
            "anchor_age": [65, 50, 17],
            "anchor_year": [2200, 2200, 2200],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [11, 12, 21, 31],
            "admittime": pd.to_datetime(
                ["2200-01-01", "2200-03-01", "2200-01-01", "2200-01-01"]
            ),
            "dischtime": pd.to_datetime(
                ["2200-01-02", "2200-03-05", "2200-01-05", "2200-01-05"]
            ),
            "hospital_expire_flag": [0, 1, 1, 0],
        }
    )
    icustays = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [11, 12, 21, 31],
            "stay_id": [101, 102, 201, 301],
            "intime": pd.to_datetime(
                ["2200-01-01", "2200-03-01", "2200-01-01", "2200-01-01"]
            ),
            "outtime": pd.to_datetime(
                ["2200-01-02", "2200-03-04", "2200-01-04", "2200-01-04"]
            ),
            "los": [1.0, 3.0, 3.0, 3.0],
        }
    )

    cohort = build_formal_cohort(patients, admissions, icustays)

    # Subject 1 is excluded: their first ICU stay is short; the later long stay
    # must not silently become their "first" episode. Subject 3 is a minor.
    assert len(cohort) == 1
    assert cohort.iloc[0]["Outcome"] == 1
    assert cohort.iloc[0]["RecordID"] == "E000001"
    assert cohort.iloc[0]["PatientID"] == "P000001"


def test_event_conflict_policy_prefers_labevents_then_latest_timestamp():
    old_time = pd.Timestamp("2200-01-01 01:00:00").value
    new_time = pd.Timestamp("2200-01-01 02:00:00").value

    assert should_replace_event(0, old_time, 1, old_time)
    assert not should_replace_event(1, old_time, 0, new_time)
    assert should_replace_event(1, old_time, 1, new_time)
    assert not should_replace_event(1, new_time, 1, old_time)


def test_formal_config_has_no_smoke_limits():
    config_path = Path("configs/mimiciv_formal.json")
    config = pd.read_json(config_path, typ="series")

    assert bool(config["full_scan"]) is True
    assert config["cohort_limit"] is None
    assert config["max_chartevents_rows"] is None
    assert config["observation_hours"] == 48
    assert config["time_bin_hours"] == 1
