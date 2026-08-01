"""Dependency-light ColaCare roles for fictional end-to-end smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from utils.agent_response_contract import (
    DISCUSSION_RESPONSE_SCHEMA,
    validate_discussion_response,
)
from utils.llm_client import LLMClient, LLMError, parse_json_object
from utils.local_rag import retrieve_lexically
from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction


class SmokePipelineError(RuntimeError):
    """Raised when a fictional smoke stage cannot produce a valid result."""


DISCUSSION_SCHEMA = DISCUSSION_RESPONSE_SCHEMA


def validate_synthetic_patient(patient: Mapping[str, Any]) -> dict[str, Any]:
    direct_identifiers = {"name", "mrn", "ssn", "address", "date_of_birth", "dob", "email", "phone"}
    found = direct_identifiers.intersection(key.lower() for key in patient)
    if found:
        raise SmokePipelineError("synthetic fixture contains a prohibited direct identifier field")
    if patient.get("synthetic") is not True:
        raise SmokePipelineError("smoke fixture must declare synthetic=true")
    patient_id = patient.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.startswith("SYNTHETIC-"):
        raise SmokePipelineError("patient_id must start with SYNTHETIC-")
    age = patient.get("age")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not 0 <= age <= 120:
        raise SmokePipelineError("age must be a fictional numeric value between 0 and 120")
    if not isinstance(patient.get("diagnoses"), list) or not patient["diagnoses"]:
        raise SmokePipelineError("diagnoses must be a non-empty list")
    for field in ("vital_signs", "laboratory_results", "ehr_expert"):
        if not isinstance(patient.get(field), dict) or not patient[field]:
            raise SmokePipelineError(f"{field} must be a non-empty object")
    expert_probability = patient["ehr_expert"].get("risk_probability")
    if (
        isinstance(expert_probability, bool)
        or not isinstance(expert_probability, (int, float))
        or not 0 <= float(expert_probability) <= 1
    ):
        raise SmokePipelineError("ehr_expert.risk_probability must be between 0 and 1")
    return dict(patient)


class SyntheticDoctorAgent:
    """One sequential DoctorAgent role operating only on fictional input."""

    def __init__(
        self,
        client: LLMClient,
        agent_id: str,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.client = client
        self.agent_id = agent_id
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds

    def analyze(
        self,
        patient: Mapping[str, Any],
        evidence: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        patient = validate_synthetic_patient(patient)
        evidence_text = ""
        if evidence:
            formatted = [
                f'Document [{item["id"]}] ({item["title"]}): {item["content"]}'
                for item in evidence
            ]
            evidence_text = "\nRelevant fictional documents:\n" + "\n".join(formatted)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a ColaCare DoctorAgent evaluating entirely fictional test data. "
                    "Use only the supplied record. Return only the requested JSON object. "
                    "Do not provide hidden chain-of-thought; give one short auditable conclusion."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/no_think\nAgent ID: {self.agent_id}\nFictional EHR record:\n"
                    + json.dumps(patient, ensure_ascii=False, sort_keys=True)
                    + evidence_text
                    + "\nEstimate test risk. This is software validation, not clinical advice."
                ),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat(messages, response_schema=PREDICTION_SCHEMA)
                prediction = validate_prediction(parse_json_object(response.content))
                return {
                    "agent_id": self.agent_id,
                    "role": "DoctorAgent",
                    **prediction,
                    "attempts": attempt,
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    },
                }
            except (LLMError, SmokePipelineError) as exc:
                last_error = exc
                if attempt < self.max_attempts and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        raise SmokePipelineError(
            f"DoctorAgent failed to return valid JSON after {self.max_attempts} attempts "
            f"({type(last_error).__name__})"
        ) from last_error

    def discuss(
        self,
        patient: Mapping[str, Any],
        meta_review: Mapping[str, Any],
        evidence: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        patient = validate_synthetic_patient(patient)
        evidence_text = ""
        if evidence:
            evidence_text = "\n".join(
                f'Document [{item["id"]}] ({item["title"]}): {item["content"]}'
                for item in evidence
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a DoctorAgent in exactly one fictional consultation round. "
                    "Return only JSON, no hidden chain-of-thought, and one short critique."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/no_think\nAgent ID: {self.agent_id}\nFictional patient: "
                    + json.dumps(patient, ensure_ascii=False, sort_keys=True)
                    + "\nMetaAgent result: "
                    + json.dumps(meta_review, ensure_ascii=False, sort_keys=True)
                    + ("\nFictional evidence:\n" + evidence_text if evidence_text else "")
                    + "\nState agree/disagree, a revised probability, and a concise summary."
                ),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat(messages, response_schema=DISCUSSION_SCHEMA)
                value = _validate_discussion(parse_json_object(response.content))
                return {
                    "agent_id": self.agent_id,
                    "role": "DoctorAgent",
                    **value,
                    "attempts": attempt,
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    },
                }
            except (LLMError, SmokePipelineError) as exc:
                last_error = exc
                if attempt < self.max_attempts and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        raise SmokePipelineError(
            f"DoctorAgent discussion failed after {self.max_attempts} attempts "
            f"({type(last_error).__name__})"
        ) from last_error


class SyntheticMetaAgent:
    """MetaAgent role for combining fictional DoctorAgent outputs."""

    def __init__(self, client: LLMClient, max_attempts: int = 3) -> None:
        self.client = client
        self.max_attempts = max_attempts

    def summarize(
        self,
        patient: Mapping[str, Any],
        doctor_reviews: list[Mapping[str, Any]],
        prior_meta: Mapping[str, Any] | None = None,
        critiques: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        patient = validate_synthetic_patient(patient)
        payload = {
            "patient_id": patient["patient_id"],
            "ehr_expert_probability": patient["ehr_expert"]["risk_probability"],
            "doctor_reviews": doctor_reviews,
        }
        if prior_meta is not None:
            payload["prior_meta"] = prior_meta
        if critiques is not None:
            payload["doctor_critiques"] = critiques
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the ColaCare MetaAgent for fictional software testing. "
                    "Synthesize supplied DoctorAgent outputs. Return only JSON and one concise "
                    "auditable conclusion, without hidden chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": "/no_think\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat(messages, response_schema=PREDICTION_SCHEMA)
                prediction = validate_prediction(parse_json_object(response.content))
                return {
                    "agent_id": "meta-1",
                    "role": "MetaAgent",
                    **prediction,
                    "attempts": attempt,
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    },
                }
            except LLMError as exc:
                last_error = exc
        raise SmokePipelineError(
            f"MetaAgent failed after {self.max_attempts} attempts ({type(last_error).__name__})"
        ) from last_error


def _validate_discussion(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_discussion_response(value)
    except LLMError as exc:
        raise SmokePipelineError(str(exc)) from exc


def _prediction_from(review: Mapping[str, Any]) -> dict[str, Any]:
    return {key: review[key] for key in ("risk_probability", "prediction", "reasoning_summary")}


def _run_two_doctors(
    patient: Mapping[str, Any],
    client: LLMClient,
    evidence: list[Mapping[str, Any]] | None = None,
) -> tuple[list[SyntheticDoctorAgent], list[dict[str, Any]]]:
    agents = [SyntheticDoctorAgent(client, "doctor-1"), SyntheticDoctorAgent(client, "doctor-2")]
    reviews = [agent.analyze(patient, evidence=evidence) for agent in agents]
    return agents, reviews


def run_single_stage(patient: Mapping[str, Any], client: LLMClient) -> dict[str, Any]:
    patient = validate_synthetic_patient(patient)
    doctor = SyntheticDoctorAgent(client, "doctor-1")
    review = doctor.analyze(patient)
    final_prediction = {
        key: review[key]
        for key in ("risk_probability", "prediction", "reasoning_summary")
    }
    return {
        "stage": "single",
        "synthetic_patient_id": patient["patient_id"],
        "doctor_count": 1,
        "meta_agent_enabled": False,
        "discussion_rounds": 0,
        "rag_enabled": False,
        "doctor_results": [review],
        "final_prediction": final_prediction,
        "disclaimer": "Fictional software smoke test only; not clinical advice.",
    }


def run_two_doctor_stage(patient: Mapping[str, Any], client: LLMClient) -> dict[str, Any]:
    patient = validate_synthetic_patient(patient)
    _, reviews = _run_two_doctors(patient, client)
    probability = sum(review["risk_probability"] for review in reviews) / len(reviews)
    return {
        "stage": "two-doctors",
        "synthetic_patient_id": patient["patient_id"],
        "doctor_count": 2,
        "meta_agent_enabled": False,
        "discussion_rounds": 0,
        "rag_enabled": False,
        "doctor_results": reviews,
        "final_prediction": {
            "risk_probability": probability,
            "prediction": int(probability >= 0.5),
            "reasoning_summary": "Deterministic average of two sequential fictional DoctorAgent outputs.",
        },
        "disclaimer": "Fictional software smoke test only; not clinical advice.",
    }


def run_meta_stage(patient: Mapping[str, Any], client: LLMClient) -> dict[str, Any]:
    patient = validate_synthetic_patient(patient)
    _, reviews = _run_two_doctors(patient, client)
    meta = SyntheticMetaAgent(client).summarize(patient, reviews)
    return {
        "stage": "meta",
        "synthetic_patient_id": patient["patient_id"],
        "doctor_count": 2,
        "meta_agent_enabled": True,
        "discussion_rounds": 0,
        "rag_enabled": False,
        "doctor_results": reviews,
        "meta_results": [meta],
        "final_prediction": _prediction_from(meta),
        "disclaimer": "Fictional software smoke test only; not clinical advice.",
    }


def _run_discussion(
    patient: Mapping[str, Any],
    client: LLMClient,
    evidence: list[Mapping[str, Any]] | None,
    stage: str,
) -> dict[str, Any]:
    agents, reviews = _run_two_doctors(patient, client, evidence=evidence)
    meta_agent = SyntheticMetaAgent(client)
    initial_meta = meta_agent.summarize(patient, reviews)
    critiques = [agent.discuss(patient, initial_meta, evidence=evidence) for agent in agents]
    revised_meta = meta_agent.summarize(
        patient, reviews, prior_meta=initial_meta, critiques=critiques
    )
    return {
        "stage": stage,
        "synthetic_patient_id": patient["patient_id"],
        "doctor_count": 2,
        "meta_agent_enabled": True,
        "discussion_rounds": 1,
        "rag_enabled": evidence is not None,
        "doctor_results": reviews,
        "meta_results": [initial_meta, revised_meta],
        "discussion": {"round": 1, "doctor_critiques": critiques},
        "final_prediction": _prediction_from(revised_meta),
        "disclaimer": "Fictional software smoke test only; not clinical advice.",
    }


def run_discussion_stage(patient: Mapping[str, Any], client: LLMClient) -> dict[str, Any]:
    patient = validate_synthetic_patient(patient)
    return _run_discussion(patient, client, evidence=None, stage="discussion")


def run_rag_stage(
    patient: Mapping[str, Any],
    client: LLMClient,
    documents: list[Mapping[str, Any]],
) -> dict[str, Any]:
    patient = validate_synthetic_patient(patient)
    query = " ".join(
        list(map(str, patient["diagnoses"]))
        + list(patient["vital_signs"].keys())
        + list(patient["laboratory_results"].keys())
    )
    evidence = retrieve_lexically(query, documents, top_k=2)
    result = _run_discussion(patient, client, evidence=evidence, stage="rag")
    result["retrieval_backend"] = "synthetic-lexical-overlap"
    result["retrieved_documents"] = evidence
    return result


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
