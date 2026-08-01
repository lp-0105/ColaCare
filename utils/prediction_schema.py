"""Shared structured prediction schema for local smoke tests."""

from __future__ import annotations

from typing import Any, Mapping

from utils.agent_response_contract import (
    PREDICTION_RESPONSE_SCHEMA,
    validate_prediction_response,
)


PREDICTION_SCHEMA = PREDICTION_RESPONSE_SCHEMA


def validate_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    return validate_prediction_response(value)
