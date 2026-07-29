from __future__ import annotations

import pytest

from utils.privacy_validator import PrivacyViolation, validate_privacy


@pytest.mark.parametrize(
    "payload",
    [
        {"PatientID": "P000001"},
        {"nested": {"subject_id": 123}},
        {"label": 1},
        {"outcome": 0},
        {"embeddings": [0.1, 0.2]},
        {"note": "measured at 2148-01-01 12:34:56"},
        {"record": "hadm_id=123456"},
    ],
)
def test_privacy_validator_rejects_identifiers_labels_embeddings_and_timestamps(payload):
    with pytest.raises(PrivacyViolation):
        validate_privacy(payload)


def test_privacy_validator_accepts_anonymous_contract_content():
    validate_privacy(
        {
            "anonymous_sample_id": "sample_000000",
            "split": "test",
            "probability": 0.35,
            "time_hour": 12,
            "age_band": "65-74",
        }
    )
