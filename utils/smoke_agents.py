"""Dependency-light ColaCare roles for fictional end-to-end smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from utils.llm_client import LLMClient, LLMError, parse_json_object
from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction


class SmokePipelineError(RuntimeError):
    """Raised when a fictional smoke stage cannot produce a valid result."""


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

    def analyze(self, patient: Mapping[str, Any]) -> dict[str, Any]:
        patient = validate_synthetic_patient(patient)
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
                    "/no_think\nFictional EHR record:\n"
                    + json.dumps(patient, ensure_ascii=False, sort_keys=True)
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
            except LLMError as exc:
                last_error = exc
                if attempt < self.max_attempts and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        raise SmokePipelineError(
            f"DoctorAgent failed to return valid JSON after {self.max_attempts} attempts "
            f"({type(last_error).__name__})"
        ) from last_error


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
