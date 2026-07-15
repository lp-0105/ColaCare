import json
from pathlib import Path
import tempfile
import unittest

from utils.llm_client import LLMResponse
from utils.smoke_agents import (
    SmokePipelineError,
    SyntheticDoctorAgent,
    run_single_stage,
    validate_synthetic_patient,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_patient.json"


def valid_response(probability=0.42, prediction=0):
    return LLMResponse(
        json.dumps(
            {
                "risk_probability": probability,
                "prediction": prediction,
                "reasoning_summary": "Fictional oxygen and respiratory findings support this test result.",
            }
        ),
        prompt_tokens=100,
        completion_tokens=40,
    )


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, response_schema=None):
        self.calls.append({"messages": messages, "schema": response_schema})
        return self.responses.pop(0)


class SyntheticPatientTests(unittest.TestCase):
    def test_committed_fixture_is_explicitly_fictional_and_complete(self):
        patient = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = validate_synthetic_patient(patient)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["patient_id"], "SYNTHETIC-0001")
        self.assertIn("vital_signs", result)
        self.assertIn("laboratory_results", result)
        self.assertIn("ehr_expert", result)

    def test_fixture_with_direct_identifier_field_is_rejected(self):
        patient = json.loads(FIXTURE.read_text(encoding="utf-8"))
        patient["name"] = "Not Allowed"
        with self.assertRaisesRegex(SmokePipelineError, "identifier"):
            validate_synthetic_patient(patient)


class SyntheticDoctorAgentTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_single_agent_returns_validated_structured_prediction(self):
        client = QueueClient([valid_response()])
        result = SyntheticDoctorAgent(client, "doctor-1", retry_delay_seconds=0).analyze(
            self.patient
        )
        self.assertEqual(result["role"], "DoctorAgent")
        self.assertEqual(result["risk_probability"], 0.42)
        self.assertEqual(result["attempts"], 1)
        prompt = client.calls[0]["messages"][-1]["content"]
        self.assertIn("SYNTHETIC-0001", prompt)
        self.assertNotIn("Document [", prompt)

    def test_invalid_first_response_is_retried_once(self):
        client = QueueClient([LLMResponse("not JSON"), valid_response()])
        result = SyntheticDoctorAgent(client, "doctor-1", retry_delay_seconds=0).analyze(
            self.patient
        )
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(client.calls), 2)

    def test_retries_are_bounded(self):
        client = QueueClient([LLMResponse("bad"), LLMResponse("bad"), LLMResponse("bad")])
        with self.assertRaisesRegex(SmokePipelineError, "3 attempts"):
            SyntheticDoctorAgent(client, "doctor-1", retry_delay_seconds=0).analyze(
                self.patient
            )

    def test_single_stage_disables_meta_discussion_and_rag(self):
        client = QueueClient([valid_response()])
        result = run_single_stage(self.patient, client)
        self.assertEqual(result["doctor_count"], 1)
        self.assertFalse(result["meta_agent_enabled"])
        self.assertEqual(result["discussion_rounds"], 0)
        self.assertFalse(result["rag_enabled"])
        self.assertEqual(result["final_prediction"]["prediction"], 0)


class ArtifactTests(unittest.TestCase):
    def test_json_write_is_atomic_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json_atomic(path, {"synthetic": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"synthetic": True})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
