from __future__ import annotations

from typing import Any

import numpy as np


def _validate_inputs(
    raw_x: np.ndarray,
    missing_mask: np.ndarray,
    feature_names: list[str],
    observation_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_x, dtype=np.float64)
    mask = np.asarray(missing_mask, dtype=np.float64)
    expected = (observation_hours, len(feature_names))
    if raw.shape != expected or mask.shape != expected:
        raise ValueError(
            f"raw_x and missing_mask shape must both be {expected}; "
            f"got {raw.shape} and {mask.shape}"
        )
    if not np.isfinite(mask).all() or not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError("missing_mask must contain only finite 0/1 values")
    observed = mask == 0
    if not np.isfinite(raw[observed]).all():
        raise ValueError("raw_x contains a non-finite observed value")
    if np.isinf(raw).any():
        raise ValueError("raw_x contains infinity")
    return raw, mask


def _age_band(age: float, config: dict[str, Any]) -> str:
    for band in config["age_bands"]:
        if float(band["minimum"]) <= age < float(band["maximum_exclusive"]):
            return str(band["label"])
    return "outside configured adult bands"


def _direction(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "increasing"
    if delta < -tolerance:
        return "decreasing"
    return "stable"


def _static_summary(
    raw: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sex_index = feature_names.index("Sex")
    age_index = feature_names.index("Age")
    sex_hours = np.flatnonzero(mask[:, sex_index] == 0)
    age_hours = np.flatnonzero(mask[:, age_index] == 0)

    if len(sex_hours):
        raw_sex = raw[int(sex_hours[0]), sex_index]
        integer_sex = int(round(float(raw_sex)))
        if abs(float(raw_sex) - integer_sex) > 1e-6:
            raise ValueError("Sex must use the configured integer coding")
        sex_value = config["sex_mapping"].get(str(integer_sex))
        if sex_value is None:
            raise ValueError(f"Sex value {integer_sex} is outside the configured mapping")
        sex = {"value": sex_value, "observed_status": "observed"}
    else:
        sex = {"value": None, "observed_status": "missing"}

    if len(age_hours):
        age_value = float(raw[int(age_hours[0]), age_index])
        age = {
            "age_band": _age_band(age_value, config),
            "observed_status": "observed",
        }
    else:
        age = {"age_band": None, "observed_status": "missing"}
    return age, sex


def _continuous_summary(
    raw: np.ndarray,
    mask: np.ndarray,
    index: int,
    config: dict[str, Any],
    unit: str | None,
) -> dict[str, Any]:
    observed_hours = np.flatnonzero(mask[:, index] == 0)
    missing_fraction = float(np.mean(mask[:, index] != 0))
    if not len(observed_hours):
        return {
            "observed_status": "missing",
            "first_observed": None,
            "latest_observed": None,
            "min_observed": None,
            "max_observed": None,
            "trend": {
                "method": config["trend"]["method"],
                "delta": None,
                "direction": "not_available",
            },
            "observed_count": 0,
            "missing_fraction": missing_fraction,
            "unit": unit,
        }
    values = raw[observed_hours, index]
    first = float(values[0])
    latest = float(values[-1])
    delta = latest - first
    return {
        "observed_status": "observed",
        "first_observed": first,
        "latest_observed": latest,
        "min_observed": float(values.min()),
        "max_observed": float(values.max()),
        "trend": {
            "method": config["trend"]["method"],
            "delta": float(delta),
            "direction": _direction(
                delta, float(config["trend"]["zero_tolerance"])
            ),
        },
        "observed_count": int(len(observed_hours)),
        "missing_fraction": missing_fraction,
        "unit": unit,
    }


def _categorical_summary(
    raw: np.ndarray,
    mask: np.ndarray,
    indices: list[int],
    feature_names: list[str],
) -> dict[str, Any]:
    group_mask = mask[:, indices]
    row_missing = np.all(group_mask == 1, axis=1)
    row_observed = np.all(group_mask == 0, axis=1)
    if not np.all(row_missing | row_observed):
        raise ValueError("categorical one-hot group has inconsistent missing-mask values")
    observed_hours = np.flatnonzero(row_observed)
    states: list[str] = []
    for hour in observed_hours:
        values = raw[int(hour), indices]
        active = np.flatnonzero(np.isclose(values, 1.0))
        if len(active) != 1:
            raise ValueError(
                "observed categorical group must have exactly one active level"
            )
        name = feature_names[indices[int(active[0])]]
        states.append(name.split("->", maxsplit=1)[1])
    if not states:
        return {
            "observed_status": "missing",
            "first_observed_state": None,
            "latest_observed_state": None,
            "observed_state_changes": 0,
            "observed_count": 0,
            "missing_fraction": 1.0,
        }
    changes = sum(left != right for left, right in zip(states, states[1:]))
    return {
        "observed_status": "observed",
        "first_observed_state": states[0],
        "latest_observed_state": states[-1],
        "observed_state_changes": int(changes),
        "observed_count": int(len(states)),
        "missing_fraction": float(np.mean(row_missing)),
    }


def build_patient_summary(
    raw_x: np.ndarray,
    missing_mask: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, privacy-reduced summary from raw values and mask."""

    contract = config["feature_contract"]
    expected_names = list(contract["feature_names"])
    if list(feature_names) != expected_names:
        raise ValueError("feature_names do not match the formal_v2 61-column contract")
    raw, mask = _validate_inputs(
        raw_x,
        missing_mask,
        feature_names,
        int(config["observation_window_hours"]),
    )
    age, sex = _static_summary(raw, mask, feature_names, config)
    units = contract["units"]
    continuous = {
        name: _continuous_summary(
            raw,
            mask,
            feature_names.index(name),
            config,
            units.get(name),
        )
        for name in contract["continuous_features"]
    }
    categorical = {}
    for group_name, columns in contract["categorical_groups"].items():
        indices = [feature_names.index(column) for column in columns]
        categorical[group_name] = _categorical_summary(
            raw, mask, indices, feature_names
        )

    dynamic_mask = mask[:, 2:]
    return {
        "age": age,
        "sex": sex,
        "continuous_features": continuous,
        "categorical_features": categorical,
        "data_completeness": {
            "dynamic_observed_cells": int(np.count_nonzero(dynamic_mask == 0)),
            "dynamic_total_cells": int(dynamic_mask.size),
            "dynamic_missing_fraction": float(np.mean(dynamic_mask != 0)),
        },
        "missingness_semantics": (
            "missing raw cells may be imputed for model input; imputed values are "
            "never reported as observed clinical measurements"
        ),
    }
