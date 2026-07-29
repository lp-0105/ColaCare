from __future__ import annotations

from typing import Any

import numpy as np


TEMPORAL_IMPORTANCE_SEMANTICS = (
    "native time-by-feature model importance; values may be signed; "
    "ranked by absolute magnitude summed over time; within-model ranking only"
)
CONCARE_IMPORTANCE_SEMANTICS = (
    "unsigned final cross-channel attention; time already aggregated; "
    "within-model ranking only"
)


def _feature_group(feature_name: str) -> str:
    if feature_name in {"Sex", "Age"}:
        return "demographics"
    if "->" in feature_name:
        return f"categorical:{feature_name.split('->', maxsplit=1)[0]}"
    return "continuous"


def _observed_raw_value(
    raw_x: np.ndarray,
    missing_mask: np.ndarray,
    feature_names: list[str],
    feature_index: int,
    hour: int,
    config: dict[str, Any],
) -> tuple[str, Any, str]:
    if missing_mask[hour, feature_index] != 0:
        return "missing", None, "imputed"
    value = float(raw_x[hour, feature_index])
    feature_name = feature_names[feature_index]
    if feature_name == "Sex":
        rendered: Any = config["sex_mapping"].get(str(int(round(value))), "unknown")
    elif feature_name == "Age":
        rendered = next(
            (
                band["label"]
                for band in config["age_bands"]
                if float(band["minimum"])
                <= value
                < float(band["maximum_exclusive"])
            ),
            "outside configured adult bands",
        )
    elif "->" in feature_name:
        rendered = feature_name.split("->", maxsplit=1)[1] if value == 1 else "not_active"
    else:
        rendered = value
    return "observed", rendered, "observed"


def build_temporal_evidence(
    *,
    model_name: str,
    importance: np.ndarray,
    raw_x: np.ndarray,
    missing_mask: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    values = np.asarray(importance, dtype=np.float64)
    if values.shape != (48, 61):
        raise ValueError(f"{model_name} importance shape must be (48, 61)")
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} importance must be finite")
    scores = np.abs(values).sum(axis=0)
    ordered = sorted(range(61), key=lambda index: (-scores[index], index))
    evidence = []
    units = config["feature_contract"]["units"]
    for feature_index in ordered[:top_k]:
        hour = int(np.argmax(np.abs(values[:, feature_index])))
        observed_status, raw_value, model_input_status = _observed_raw_value(
            raw_x,
            missing_mask,
            feature_names,
            feature_index,
            hour,
            config,
        )
        feature_name = feature_names[feature_index]
        evidence.append(
            {
                "feature_name": feature_name,
                "feature_group": _feature_group(feature_name),
                "time_hour": hour,
                "importance": float(values[hour, feature_index]),
                "ranking_score": float(scores[feature_index]),
                "signed_or_absolute": (
                    "native value may be signed; ranking_score is the absolute "
                    "importance summed across 48 hours"
                ),
                "observed_status": observed_status,
                "model_input_status": model_input_status,
                "raw_value": raw_value,
                "unit": units.get(feature_name),
            }
        )
    return {
        "importance_semantics": TEMPORAL_IMPORTANCE_SEMANTICS,
        "top_k": top_k,
        "top_evidence": evidence,
    }


def build_concare_evidence(
    *,
    importance: np.ndarray,
    raw_x: np.ndarray,
    missing_mask: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    mapping: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    values = np.asarray(importance, dtype=np.float64)
    if values.shape != (60,):
        raise ValueError("ConCare importance shape must be (60,)")
    if not np.isfinite(values).all():
        raise ValueError("ConCare importance must be finite")
    if float(values.min()) < -1e-7:
        raise ValueError("ConCare attention must be non-negative")
    if not np.isclose(float(values.sum()), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("ConCare attention must sum to one in eval mode")
    if mapping.get("importance_semantics") != CONCARE_IMPORTANCE_SEMANTICS:
        raise ValueError("ConCare importance semantics mismatch")
    dimensions = mapping.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 60:
        raise ValueError("ConCare mapping must contain exactly 60 dimensions")
    if [item.get("index") for item in dimensions] != list(range(60)):
        raise ValueError("ConCare mapping indices must be contiguous 0..59")
    if [item["feature_name"] for item in dimensions[:59]] != feature_names[2:]:
        raise ValueError("ConCare dynamic feature ordering does not match formal_v2")
    if dimensions[59].get("source_input_indices") != [0, 1]:
        raise ValueError("ConCare final dimension must be the joint demographics context")

    ordered = sorted(range(60), key=lambda index: (-abs(values[index]), index))
    units = config["feature_contract"]["units"]
    evidence = []
    for index in ordered[:top_k]:
        dimension = dimensions[index]
        source_indices = list(dimension["source_input_indices"])
        if index == 59:
            observed_status = (
                "observed_within_window"
                if all(np.any(missing_mask[:, item] == 0) for item in source_indices)
                else "partially_or_fully_missing"
            )
            model_input_status = (
                "observed"
                if all(np.all(missing_mask[:, item] == 0) for item in source_indices)
                else "mixed_observed_and_imputed"
            )
        else:
            source_index = source_indices[0]
            observed = missing_mask[:, source_index] == 0
            observed_status = (
                "observed_within_window" if np.any(observed) else "missing_entire_window"
            )
            model_input_status = (
                "observed"
                if np.all(observed)
                else ("imputed_only" if not np.any(observed) else "mixed_observed_and_imputed")
            )
        feature_name = dimension["feature_name"]
        evidence.append(
            {
                "feature_name": feature_name,
                "feature_group": dimension["feature_group"],
                "time_hour": None,
                "importance": float(values[index]),
                "ranking_score": float(abs(values[index])),
                "signed_or_absolute": "unsigned softmax attention weight",
                "observed_status": observed_status,
                "model_input_status": model_input_status,
                "raw_value": None,
                "raw_value_semantics": (
                    "not rendered because the exported attention is already "
                    "aggregated over the 48-hour window"
                ),
                "unit": units.get(feature_name),
            }
        )
    return {
        "importance_semantics": CONCARE_IMPORTANCE_SEMANTICS,
        "top_k": top_k,
        "top_evidence": evidence,
    }
