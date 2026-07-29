from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


class PrivacyViolation(ValueError):
    """Raised when an Agent payload contains forbidden identity or label material."""


_FORBIDDEN_KEYS = {
    "patientid",
    "recordid",
    "subjectid",
    "hadmid",
    "stayid",
    "recordtime",
    "timestamp",
    "outcome",
    "label",
    "labels",
    "y",
    "embedding",
    "embeddings",
    "testmetrics",
}
_IDENTIFIER_TEXT = re.compile(
    r"\b(?:patient|record|subject|hadm|stay)[_-]?id\b\s*[:=]?",
    flags=re.IGNORECASE,
)
_MIMIC_STYLE_VALUE = re.compile(r"\b[PE]\d{6,}\b")
_TIMESTAMP_TEXT = re.compile(
    r"\b(?:19|20|21)\d{2}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"
)
_ANONYMOUS_SAMPLE = re.compile(r"^sample_\d{6}$")


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _validate_text(value: str, path: str) -> None:
    if _ANONYMOUS_SAMPLE.fullmatch(value):
        return
    if _IDENTIFIER_TEXT.search(value):
        raise PrivacyViolation(f"identifier-like text at {path}")
    if _MIMIC_STYLE_VALUE.search(value):
        raise PrivacyViolation(f"database-style identifier value at {path}")
    if _TIMESTAMP_TEXT.search(value):
        raise PrivacyViolation(f"real timestamp-like value at {path}")


def validate_privacy(value: Any, path: str = "$") -> None:
    """Recursively reject identifiers, targets, embeddings, and real timestamps."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_KEYS:
                raise PrivacyViolation(f"forbidden key {key!r} at {path}")
            validate_privacy(item, f"{path}.{key}")
        return
    if isinstance(value, str):
        _validate_text(value, path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            validate_privacy(item, f"{path}[{index}]")
