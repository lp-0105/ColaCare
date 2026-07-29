from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_expert_agent_inputs import main as build_main
from scripts.generate_expert_agent_synthetic_demo import main as demo_main
from scripts.validate_expert_agent_inputs import main as validate_main
from tests.adapter_fixtures import synthetic_case
from utils.expert_agent_adapter import build_expert_agent_input
from utils.mimiciv_preprocess_utils import formal_dynamic_columns
from utils.privacy_validator import validate_privacy


def _build(case: dict, *, top_k: int = 3) -> dict:
    return build_expert_agent_input(
        raw_x=case["raw_x"],
        missing_mask=case["missing_mask"],
        expert_outputs=case["expert_outputs"],
        anonymous_sample_id="sample_000000",
        split="test",
        config=case["config"],
        concare_mapping=case["concare_mapping"],
        top_k=top_k,
        synthetic=True,
    )


def test_agent_input_whitelists_expert_fields_and_preserves_probabilities():
    case = synthetic_case()
    payload = _build(case)
    assert payload["schema_version"] == "expert-agent-input-v1"
    assert payload["anonymous_sample_id"] == "sample_000000"
    assert {
        name: value["probability"]
        for name, value in payload["expert_predictions"].items()
    } == {"RETAIN": 0.31, "ConCare": 0.34, "AdaCare": 0.36}
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in ("labels", "outcome", "embeddings", "patientid", "recordid"):
        assert forbidden not in serialized
    validate_privacy(payload)


def test_temporal_top_k_uses_absolute_sum_and_maps_hour_and_missingness():
    case = synthetic_case()
    payload = _build(case, top_k=3)
    retain = payload["expert_evidence"]["RETAIN"]["top_evidence"]
    assert [item["feature_name"] for item in retain] == [
        "Heart Rate",
        "Glucose",
        "pH",
    ]
    assert retain[0]["time_hour"] == 47
    assert retain[0]["raw_value"] == 110.0
    assert retain[0]["observed_status"] == "observed"
    assert retain[2]["observed_status"] == "missing"
    assert retain[2]["raw_value"] is None
    assert retain[2]["model_input_status"] == "imputed"


def test_concare_mapping_is_exact_and_does_not_invent_a_time_point():
    case = synthetic_case()
    mapping = case["concare_mapping"]
    assert len(mapping["dimensions"]) == 60
    assert [item["index"] for item in mapping["dimensions"]] == list(range(60))
    assert [item["feature_name"] for item in mapping["dimensions"][:59]] == (
        case["feature_names"][2:]
    )
    assert mapping["dimensions"][59]["feature_name"] == "Demographics context (Sex + Age)"

    payload = _build(case, top_k=3)
    evidence = payload["expert_evidence"]["ConCare"]
    assert evidence["importance_semantics"] == (
        "unsigned final cross-channel attention; time already aggregated; "
        "within-model ranking only"
    )
    assert all(item["time_hour"] is None for item in evidence["top_evidence"])
    assert evidence["top_evidence"][2]["feature_group"] == "demographics_aggregate"
    assert evidence["top_evidence"][2]["raw_value"] is None


def test_machine_readable_mappings_match_the_formal_v2_source_contract():
    case = synthetic_case()
    dynamic = formal_dynamic_columns()
    assert case["feature_names"] == ["Sex", "Age", *dynamic]
    assert [
        item["feature_name"]
        for item in case["concare_mapping"]["dimensions"][:59]
    ] == dynamic
    assert (
        case["concare_mapping"]["inactive_compatibility_feature_names"]
        == case["config"]["feature_contract"]["inactive_compatibility_columns"]
    )
    assert len(
        case["concare_mapping"]["inactive_compatibility_feature_names"]
    ) == 18


def test_nonfinite_probability_or_importance_is_rejected():
    case = synthetic_case()
    case["expert_outputs"]["RETAIN"]["probability"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _build(case)


def test_concare_attention_must_be_unsigned_and_normalized():
    case = synthetic_case()
    case["expert_outputs"]["ConCare"]["feature_importance"][0] = -0.05
    case["expert_outputs"]["ConCare"]["feature_importance"][1] += 0.10
    with pytest.raises(ValueError, match="non-negative"):
        _build(case)

    case = synthetic_case()
    case["expert_outputs"]["ConCare"]["feature_importance"] *= 0.5
    with pytest.raises(ValueError, match="sum to one"):
        _build(case)

    case = synthetic_case()
    case["expert_outputs"]["ConCare"]["feature_importance"][0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _build(case)


def test_same_input_produces_identical_json():
    case = synthetic_case()
    first = json.dumps(_build(case), sort_keys=True, separators=(",", ":"))
    second = json.dumps(_build(case), sort_keys=True, separators=(",", ":"))
    assert first.encode("utf-8") == second.encode("utf-8")


def _write_cli_fixture(root: Path) -> tuple[Path, Path, Path]:
    case_a = synthetic_case()
    case_b = synthetic_case(disagreement=True)
    formal_dir = root / "formal_v2" / "fold_1"
    expert_dir = root / "expert_outputs_v2"
    formal_dir.mkdir(parents=True)
    pd.to_pickle(
        [case_a["raw_x"], case_b["raw_x"]],
        formal_dir / "test_raw_x.pkl",
    )
    pd.to_pickle(
        [case_a["missing_mask"], case_b["missing_mask"]],
        formal_dir / "test_missing_mask.pkl",
    )

    for model_name in ("RETAIN", "ConCare", "AdaCare"):
        model_key = model_name.lower()
        model_dir = expert_dir / "test" / model_key
        model_dir.mkdir(parents=True)
        values = [case_a["expert_outputs"][model_name], case_b["expert_outputs"][model_name]]
        np.savez_compressed(
            model_dir / "chunk_00000.npz",
            anonymous_sample_indices=np.asarray([0, 1], dtype=np.int64),
            labels=np.asarray([0, 1], dtype=np.int64),
            logits=np.asarray([item["logit"] for item in values]),
            probabilities=np.asarray([item["probability"] for item in values]),
            embeddings=np.stack([item["embeddings"] for item in values]),
            feature_importance=np.stack(
                [item["feature_importance"] for item in values]
            ),
        )
        (expert_dir / f"{model_key}_export_manifest.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "model": model_name,
                    "data_version": "formal_v2",
                    "checkpoint_sha256": values[0]["checkpoint_hash"],
                    "contains_database_ids": False,
                }
            ),
            encoding="utf-8",
        )

    feature_mapping = root / "feature_mapping.json"
    feature_mapping.write_text(
        json.dumps(
            {
                "version": case_a["config"]["feature_contract"]["mapping_version"],
                "categorical_columns": case_a["feature_names"][2:49],
                "numerical_features": case_a["feature_names"][49:],
            }
        ),
        encoding="utf-8",
    )
    return formal_dir, expert_dir, feature_mapping


def test_cli_build_and_validate_synthetic_anonymous_outputs(tmp_path: Path):
    formal_dir, expert_dir, feature_mapping = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "agent_inputs"
    assert (
        build_main(
            [
                "--formal-fold-dir",
                str(formal_dir),
                "--expert-output-dir",
                str(expert_dir),
                "--feature-mapping",
                str(feature_mapping),
                "--split",
                "test",
                "--sample-indices",
                "0,1",
                "--output-dir",
                str(output_dir),
                "--top-k",
                "3",
                "--seed",
                "42",
                "--synthetic",
            ]
        )
        == 0
    )
    assert validate_main(["--input-dir", str(output_dir)]) == 0
    assert (
        build_main(
            [
                "--output-dir",
                str(output_dir),
                "--validate-only",
            ]
        )
        == 0
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["anonymous_sample_ids"] == ["sample_000000", "sample_000001"]
    assert "sample_indices" not in json.dumps(manifest).lower()
    assert (output_dir / "sample_000000.json").exists()
    assert (output_dir / "sample_000000.txt").exists()
    assert json.loads(
        (output_dir / "validation_report.json").read_text(encoding="utf-8")
    )["status"] == "PASS"


def test_synthetic_demo_cli_generates_at_most_three_labeled_examples(tmp_path: Path):
    output_dir = tmp_path / "demo"
    assert demo_main(["--output-dir", str(output_dir)]) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"] == 3
    assert manifest["synthetic_demo"] is True
    assert len(list(output_dir.glob("sample_*.json"))) == 3
    assert "SYNTHETIC DEMO — NOT REAL PATIENT DATA" in (
        output_dir / "sample_000000.txt"
    ).read_text(encoding="utf-8")
