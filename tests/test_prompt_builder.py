from __future__ import annotations

import json

from tests.adapter_fixtures import synthetic_case
from utils.deterministic_prompt_builder import build_deterministic_prompt
from utils.expert_agent_adapter import build_expert_agent_input


def _payload(disagreement: bool = False) -> dict:
    case = synthetic_case(disagreement=disagreement)
    return build_expert_agent_input(
        raw_x=case["raw_x"],
        missing_mask=case["missing_mask"],
        expert_outputs=case["expert_outputs"],
        anonymous_sample_id="sample_000000",
        split="test",
        config=case["config"],
        concare_mapping=case["concare_mapping"],
        synthetic=True,
    )


def test_prompt_is_byte_deterministic_and_contains_fixed_limitations():
    payload = _payload()
    first = build_deterministic_prompt(payload)
    second = build_deterministic_prompt(json.loads(json.dumps(payload)))
    assert first.encode("utf-8") == second.encode("utf-8")
    assert "SYNTHETIC DEMO — NOT REAL PATIENT DATA" in first
    assert "first 48 hours" in first
    assert "Labels were not provided" in first
    assert "not clinical diagnoses" in first
    assert "identity information" in first


def test_prompt_never_contains_labels_embeddings_or_database_identifiers():
    prompt = build_deterministic_prompt(_payload()).lower()
    for forbidden in (
        "patientid",
        "recordid",
        "subject_id",
        "hadm_id",
        "stay_id",
        "embedding",
        "outcome",
        '"label":',
        "2148-",
    ):
        assert forbidden not in prompt


def test_disagreement_summary_is_deterministic():
    payload = _payload(disagreement=True)
    assert payload["agreement_summary"]["disagreement_level"] == "high"
    assert payload["agreement_summary"]["model_ranking"] == [
        "AdaCare",
        "ConCare",
        "RETAIN",
    ]
