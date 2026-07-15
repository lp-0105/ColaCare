"""Deterministic lexical retrieval for fictional smoke-test documents."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


def _validate_documents(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated = []
    for document in documents:
        if document.get("synthetic") is not True:
            raise ValueError("local RAG accepts only documents marked synthetic=true")
        document_id = document.get("id")
        if not isinstance(document_id, str) or not document_id.startswith("SYN-GUIDE-"):
            raise ValueError("synthetic guideline id must start with SYN-GUIDE-")
        if not isinstance(document.get("title"), str) or not document["title"].strip():
            raise ValueError("synthetic guideline title must be non-empty")
        if not isinstance(document.get("content"), str) or not document["content"].strip():
            raise ValueError("synthetic guideline content must be non-empty")
        validated.append(dict(document))
    return validated


def load_synthetic_guidelines(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("synthetic guideline fixture must be a non-empty JSON array")
    return _validate_documents(value)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def retrieve_lexically(
    query: str,
    documents: Sequence[Mapping[str, Any]],
    top_k: int = 2,
) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("retrieval query must be non-empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    validated = _validate_documents(documents)
    query_tokens = _tokens(query)
    scored = []
    for document in validated:
        document_tokens = _tokens(document["title"] + " " + document["content"])
        score = len(query_tokens.intersection(document_tokens))
        scored.append({**document, "score": score})
    scored.sort(key=lambda item: (-item["score"], item["id"]))
    return scored[: min(top_k, len(scored))]
