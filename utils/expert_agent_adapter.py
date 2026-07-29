from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from utils.expert_evidence_builder import (
    build_concare_evidence,
    build_temporal_evidence,
)
from utils.patient_summary_builder import build_patient_summary
from utils.privacy_validator import validate_privacy


MODEL_NAMES = ("RETAIN", "ConCare", "AdaCare")
_CHECKPOINT_HASH = re.compile(r"^[0-9a-f]{64}$")
_ANONYMOUS_SAMPLE = re.compile(r"^sample_[0-9]{6}$")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _risk_band(probability: float, config: dict[str, Any]) -> str:
    for band in config["risk_bands"]:
        if probability < float(band["upper_exclusive"]):
            return str(band["label"])
    raise ValueError("risk band configuration does not cover probability")


def _prediction(
    model_name: str,
    value: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    probability = float(value["probability"])
    logit = float(value["logit"])
    if not np.isfinite(probability) or not np.isfinite(logit):
        raise ValueError(f"{model_name} probability/logit must be finite")
    if probability < 0 or probability > 1:
        raise ValueError(f"{model_name} probability must be within [0,1]")
    checkpoint_hash = str(value["checkpoint_hash"]).lower()
    if not _CHECKPOINT_HASH.fullmatch(checkpoint_hash):
        raise ValueError(f"{model_name} checkpoint_hash must be lowercase SHA256")
    return {
        "model_name": model_name,
        "probability": probability,
        "logit": logit,
        "risk_band": _risk_band(probability, config),
        "risk_band_semantics": (
            "fixed display-only model-estimate interval; not a clinical risk grade"
        ),
        "checkpoint_hash": checkpoint_hash,
    }


def _agreement(predictions: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict:
    probabilities = np.asarray(
        [predictions[name]["probability"] for name in MODEL_NAMES], dtype=np.float64
    )
    probability_range = float(probabilities.max() - probabilities.min())
    thresholds = config["disagreement"]
    if probability_range <= float(thresholds["low_upper_inclusive"]):
        level = "low"
    elif probability_range <= float(thresholds["moderate_upper_inclusive"]):
        level = "moderate"
    else:
        level = "high"
    ranking = sorted(
        MODEL_NAMES,
        key=lambda name: (-predictions[name]["probability"], name),
    )
    return {
        "mean_probability": float(probabilities.mean()),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
        "probability_range": probability_range,
        "standard_deviation": float(probabilities.std(ddof=0)),
        "model_ranking": ranking,
        "disagreement_level": level,
        "disagreement_semantics": (
            "deterministic display rule based on probability range; no LLM judgment"
        ),
    }


def build_expert_agent_input(
    *,
    raw_x: np.ndarray,
    missing_mask: np.ndarray,
    expert_outputs: dict[str, dict[str, Any]],
    anonymous_sample_id: str,
    split: str,
    config: dict[str, Any],
    concare_mapping: dict[str, Any],
    top_k: int | None = None,
    synthetic: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Adapt formal_v2 tensors and expert outputs into a label-free Agent input."""

    if not _ANONYMOUS_SAMPLE.fullmatch(anonymous_sample_id):
        raise ValueError("anonymous_sample_id must match sample_000000")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if set(expert_outputs) != set(MODEL_NAMES):
        raise ValueError(f"expert_outputs must contain exactly {MODEL_NAMES}")
    top_k = int(top_k if top_k is not None else config["default_top_k"])
    if top_k < 1 or top_k > 60:
        raise ValueError("top_k must be between 1 and 60")

    feature_names = list(config["feature_contract"]["feature_names"])
    raw = np.asarray(raw_x, dtype=np.float64)
    mask = np.asarray(missing_mask, dtype=np.float64)
    patient_summary = build_patient_summary(raw, mask, feature_names, config)
    predictions = {
        name: _prediction(name, expert_outputs[name], config) for name in MODEL_NAMES
    }
    evidence = {
        "RETAIN": build_temporal_evidence(
            model_name="RETAIN",
            importance=expert_outputs["RETAIN"]["feature_importance"],
            raw_x=raw,
            missing_mask=mask,
            feature_names=feature_names,
            config=config,
            top_k=top_k,
        ),
        "ConCare": build_concare_evidence(
            importance=expert_outputs["ConCare"]["feature_importance"],
            raw_x=raw,
            missing_mask=mask,
            feature_names=feature_names,
            config=config,
            mapping=concare_mapping,
            top_k=top_k,
        ),
        "AdaCare": build_temporal_evidence(
            model_name="AdaCare",
            importance=expert_outputs["AdaCare"]["feature_importance"],
            raw_x=raw,
            missing_mask=mask,
            feature_names=feature_names,
            config=config,
            top_k=top_k,
        ),
    }
    payload = {
        "schema_version": config["schema_version"],
        "anonymous_sample_id": anonymous_sample_id,
        "split": split,
        "observation_window_hours": int(config["observation_window_hours"]),
        "synthetic_demo": bool(synthetic),
        "patient_summary": patient_summary,
        "expert_predictions": predictions,
        "expert_evidence": evidence,
        "agreement_summary": _agreement(predictions, config),
        "provenance": {
            "data_contract": config["data_contract"],
            "feature_mapping_version": config["feature_contract"]["mapping_version"],
            "adapter_seed": int(seed),
            "clinical_value_source": "raw_x combined with missing_mask",
            "model_input_x_rendered": False,
            "targets_provided_to_agent": False,
            "fusion_vectors_provided_to_agent": False,
            "database_identifiers_included": False,
        },
    }
    validate_privacy(payload)
    return payload
