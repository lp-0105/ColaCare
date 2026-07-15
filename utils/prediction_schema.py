"""Shared structured prediction schema for local smoke tests."""

from __future__ import annotations

from typing import Any, Mapping

from utils.llm_client import LLMResponseError


PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "prediction": {"type": "integer", "enum": [0, 1]},
        "reasoning_summary": {"type": "string", "minLength": 1, "maxLength": 600},
    },
    "required": ["risk_probability", "prediction", "reasoning_summary"],
    "additionalProperties": False,
}


def validate_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"risk_probability", "prediction", "reasoning_summary"}
    if set(value) != required:
        raise LLMResponseError("prediction JSON must contain exactly the required fields")
    probability = value["risk_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise LLMResponseError("risk_probability must be a number between 0 and 1")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise LLMResponseError("risk_probability must be between 0 and 1")
    prediction = value["prediction"]
    if isinstance(prediction, bool) or not isinstance(prediction, int) or prediction not in (0, 1):
        raise LLMResponseError("prediction must be integer 0 or 1")
    summary = value["reasoning_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise LLMResponseError("reasoning_summary must be a non-empty string")
    if len(summary) > 600:
        raise LLMResponseError("reasoning_summary must be at most 600 characters")
    return {
        "risk_probability": probability,
        "prediction": prediction,
        "reasoning_summary": summary.strip(),
    }
