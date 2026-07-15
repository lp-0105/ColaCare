"""Live contract check for an OpenAI-compatible LLM service."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.llm_client import LLMClient, LLMError, LLMResponseError, parse_json_object


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


def run_api_check(client: LLMClient) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You validate a JSON API. Return only the requested object. "
                "Do not provide hidden chain-of-thought; provide one short conclusion sentence."
            ),
        },
        {
            "role": "user",
            "content": (
                "/no_think\nReturn risk_probability 0.35, prediction 0, and a short "
                "reasoning_summary stating this is a fictional API test."
            ),
        },
    ]
    response = client.chat(messages, response_schema=PREDICTION_SCHEMA)
    result = validate_prediction(parse_json_object(response.content))
    result["usage"] = {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }
    return result


def main() -> int:
    try:
        result = run_api_check(LLMClient.from_env())
    except LLMError as exc:
        print(f"LLM API check failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"LLM API check failed with unexpected {type(exc).__name__}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
