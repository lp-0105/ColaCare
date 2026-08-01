"""Strict, auditable response contract for real expert-to-Agent runs.

The first model call performs the clinical review. If and only if that call
returns otherwise-valid JSON whose ``reasoning_summary`` exceeds the hard
Unicode-character limit, subsequent calls are JSON-format repairs. Repair
calls never receive the patient prompt and may change only the summary field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping

from utils.llm_client import LLMResponseError, parse_json_object


SUMMARY_PROMPT_TARGET_CHARS = 500
SUMMARY_MAX_CHARS = 640
MAX_SUMMARY_REPAIRS = 2

SUMMARY_OUTPUT_INSTRUCTION = (
    "Return exactly one complete valid JSON object. Keep reasoning_summary at "
    "most 500 Unicode characters and at most two complete sentences or three "
    "short sentences. Do not use Markdown or provide hidden chain-of-thought; "
    "reasoning_summary must contain only the final concise clinical reasoning "
    "summary. Put detailed evidence in the existing structured fields. The "
    "validator enforces a hard limit of 640 Unicode characters."
)

PREDICTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "prediction": {"type": "integer", "enum": [0, 1]},
        "reasoning_summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": SUMMARY_MAX_CHARS,
        },
    },
    "required": ["risk_probability", "prediction", "reasoning_summary"],
    "additionalProperties": False,
}

DISCUSSION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": ["agree", "disagree"]},
        "revised_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning_summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": SUMMARY_MAX_CHARS,
        },
    },
    "required": ["stance", "revised_probability", "reasoning_summary"],
    "additionalProperties": False,
}


class ResponseContractError(LLMResponseError):
    """The response violates the explicit Agent JSON contract."""


class ReasoningSummaryLengthError(ResponseContractError):
    """The sole contract violation is an overlong reasoning summary."""

    def __init__(self, payload: Mapping[str, Any], actual_length: int) -> None:
        self.payload = dict(payload)
        self.actual_length = actual_length
        self.maximum_length = SUMMARY_MAX_CHARS
        super().__init__(
            "reasoning_summary has "
            f"{actual_length} Unicode characters; maximum is {SUMMARY_MAX_CHARS}"
        )


class SummaryRepairExhaustedError(ResponseContractError):
    """Two bounded summary-only repairs failed validation."""

    def __init__(self, original_summary_length: int, errors: list[dict[str, Any]]) -> None:
        self.original_summary_length = original_summary_length
        self.errors = list(errors)
        super().__init__(
            "reasoning_summary repair failed after "
            f"{MAX_SUMMARY_REPAIRS} attempts; original length was "
            f"{original_summary_length}, hard limit is {SUMMARY_MAX_CHARS}"
        )


@dataclass(frozen=True)
class StructuredResponseResult:
    payload: dict[str, Any]
    original_response_sha256: str
    original_summary_length: int
    repair_attempts: int
    repair_errors: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int


_MARKDOWN_PATTERNS = (
    re.compile(r"```|`[^`]+`"),
    re.compile(r"(?m)^\s{0,3}#{1,6}\s"),
    re.compile(r"(?m)^\s*[-*+]\s+"),
    re.compile(r"\[[^\]]+\]\([^\)]+\)"),
)

_PROHIBITED_AGENT_TOKENS = (
    "patientid",
    "recordid",
    "subject_id",
    "hadm_id",
    "stay_id",
    "outcome_label",
    "ground_truth",
)


def _reject_prohibited_text(text: str, *, context: str) -> None:
    lowered = text.lower()
    found = [token for token in _PROHIBITED_AGENT_TOKENS if token in lowered]
    if found:
        raise ResponseContractError(
            f"prohibited identifier or label token in {context}: {', '.join(found)}"
        )


def _validate_summary(payload: Mapping[str, Any]) -> str:
    summary = payload.get("reasoning_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ResponseContractError("reasoning_summary must be a non-empty string")
    _reject_prohibited_text(summary, context="reasoning_summary")
    if any(pattern.search(summary) for pattern in _MARKDOWN_PATTERNS):
        raise ResponseContractError("reasoning_summary must not contain Markdown")
    if len(summary) > SUMMARY_MAX_CHARS:
        raise ReasoningSummaryLengthError(payload, len(summary))
    return summary.strip()


def validate_prediction_response(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"risk_probability", "prediction", "reasoning_summary"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ResponseContractError(
            "prediction JSON must contain exactly the required fields"
        )
    probability = value["risk_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ResponseContractError("risk_probability must be a number between 0 and 1")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ResponseContractError("risk_probability must be between 0 and 1")
    prediction = value["prediction"]
    if isinstance(prediction, bool) or not isinstance(prediction, int):
        raise ResponseContractError("prediction must be integer 0 or 1")
    if prediction not in (0, 1):
        raise ResponseContractError("prediction must be integer 0 or 1")
    return {
        "risk_probability": probability,
        "prediction": prediction,
        "reasoning_summary": _validate_summary(value),
    }


def validate_discussion_response(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"stance", "revised_probability", "reasoning_summary"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ResponseContractError(
            "discussion JSON must contain exactly the required fields"
        )
    stance = value["stance"]
    if stance not in {"agree", "disagree"}:
        raise ResponseContractError("stance must be agree or disagree")
    probability = value["revised_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ResponseContractError(
            "revised_probability must be a number between 0 and 1"
        )
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ResponseContractError("revised_probability must be between 0 and 1")
    return {
        "stance": stance,
        "revised_probability": probability,
        "reasoning_summary": _validate_summary(value),
    }


def prepare_initial_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Append the summary target without mutating the caller's prompt."""
    prepared = [dict(message) for message in messages]
    for index in range(len(prepared) - 1, -1, -1):
        if prepared[index].get("role") == "system":
            prepared[index]["content"] = (
                prepared[index].get("content", "").rstrip()
                + "\n\n"
                + SUMMARY_OUTPUT_INSTRUCTION
            )
            break
    else:
        prepared.insert(0, {"role": "system", "content": SUMMARY_OUTPUT_INSTRUCTION})
    return prepared


def build_summary_repair_messages(
    original_payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build a format-only request containing no original patient prompt."""
    summary = original_payload.get("reasoning_summary")
    if not isinstance(summary, str):
        raise ResponseContractError("repair requires a string reasoning_summary")
    actual_length = len(summary)
    return [
        {
            "role": "system",
            "content": (
                "You are a deterministic JSON format-repair utility. Do not "
                "re-evaluate the case, introduce new facts, or provide chain-of-thought."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The current reasoning_summary has {actual_length} Unicode characters. "
                f"The hard maximum is {SUMMARY_MAX_CHARS}; target no more than "
                f"{SUMMARY_PROMPT_TARGET_CHARS}. Return one complete valid JSON object. "
                "Compress only reasoning_summary to at most two complete sentences or "
                "three short sentences, without Markdown. Every other field, fact, "
                "probability, evidence, and conclusion must remain unchanged as a JSON "
                "value.\nORIGINAL_JSON:\n"
                + json.dumps(
                    dict(original_payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def validate_summary_only_change(
    original: Mapping[str, Any], repaired: Mapping[str, Any]
) -> None:
    if set(original) != set(repaired):
        raise ResponseContractError("repair changed the JSON field set")
    for key in original:
        if key != "reasoning_summary" and repaired[key] != original[key]:
            raise ResponseContractError(f"repair changed protected field: {key}")


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding=encoding) != text:
            raise FileExistsError(f"refusing to overwrite different response: {path}")
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(text)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _raw_response(client: Any, fallback: str) -> str:
    raw = getattr(client, "last_raw_response", None)
    return raw if isinstance(raw, str) else fallback


def _preserve_original_response(path: Path | None, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if path is not None:
        _atomic_write_text(path, raw)
        _atomic_write_text(Path(str(path) + ".sha256"), digest, encoding="ascii")
    return digest


def request_with_summary_repair(
    client: Any,
    messages: list[dict[str, str]],
    *,
    response_schema: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    original_response_path: Path | None = None,
    max_repairs: int = MAX_SUMMARY_REPAIRS,
) -> StructuredResponseResult:
    """Make one reasoning call and at most two summary-only repair calls."""
    if max_repairs != MAX_SUMMARY_REPAIRS:
        raise ValueError(f"max_repairs is frozen at {MAX_SUMMARY_REPAIRS}")
    responses = []
    _reject_prohibited_text(
        json.dumps(messages, ensure_ascii=False, sort_keys=True),
        context="Agent prompt",
    )
    initial = client.chat(
        prepare_initial_messages(messages), response_schema=response_schema
    )
    responses.append(initial)
    raw_initial = _raw_response(client, initial.content)
    original_sha = _preserve_original_response(original_response_path, raw_initial)
    _reject_prohibited_text(raw_initial, context="raw model response")
    parsed = parse_json_object(initial.content)
    try:
        validated = validator(parsed)
    except ReasoningSummaryLengthError as exc:
        original_payload = exc.payload
        original_length = exc.actual_length
    else:
        return StructuredResponseResult(
            payload=validated,
            original_response_sha256=original_sha,
            original_summary_length=len(validated["reasoning_summary"]),
            repair_attempts=0,
            repair_errors=[],
            prompt_tokens=initial.prompt_tokens,
            completion_tokens=initial.completion_tokens,
        )

    errors: list[dict[str, Any]] = []
    for repair_number in range(1, MAX_SUMMARY_REPAIRS + 1):
        repair_messages = build_summary_repair_messages(original_payload)
        try:
            response = client.chat(repair_messages, response_schema=response_schema)
            responses.append(response)
            repaired = parse_json_object(response.content)
            validate_summary_only_change(original_payload, repaired)
            validated = validator(repaired)
        except ReasoningSummaryLengthError as exc:
            errors.append(
                {
                    "attempt": repair_number,
                    "code": "reasoning_summary_too_long",
                    "actual_length": exc.actual_length,
                    "maximum_length": exc.maximum_length,
                }
            )
            continue
        except LLMResponseError as exc:
            code = (
                "non_summary_fields_changed"
                if "changed" in str(exc)
                else "repair_contract_error"
            )
            errors.append(
                {"attempt": repair_number, "code": code, "message": str(exc)}
            )
            continue
        return StructuredResponseResult(
            payload=validated,
            original_response_sha256=original_sha,
            original_summary_length=original_length,
            repair_attempts=repair_number,
            repair_errors=errors,
            prompt_tokens=sum(item.prompt_tokens for item in responses),
            completion_tokens=sum(item.completion_tokens for item in responses),
        )
    raise SummaryRepairExhaustedError(original_length, errors)


def request_prediction_with_summary_repair(
    client: Any,
    messages: list[dict[str, str]],
    *,
    original_response_path: Path | None = None,
) -> StructuredResponseResult:
    return request_with_summary_repair(
        client,
        messages,
        response_schema=PREDICTION_RESPONSE_SCHEMA,
        validator=validate_prediction_response,
        original_response_path=original_response_path,
    )
