"""Live contract check for an OpenAI-compatible LLM service."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.llm_client import LLMClient, LLMError, parse_json_object
from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction


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
