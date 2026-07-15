import json
import unittest
from unittest.mock import patch

from scripts.check_environment import _run, redacted_llm_environment
from scripts.test_llm_api import run_api_check, validate_prediction
from utils.llm_client import LLMResponse, LLMResponseError


class PredictionValidationTests(unittest.TestCase):
    def test_valid_prediction_is_normalized(self):
        value = validate_prediction(
            {
                "risk_probability": 0.35,
                "prediction": 0,
                "reasoning_summary": "The supplied fictional values indicate moderate risk.",
            }
        )
        self.assertEqual(value["risk_probability"], 0.35)
        self.assertEqual(value["prediction"], 0)

    def test_probability_must_be_in_range(self):
        with self.assertRaisesRegex(LLMResponseError, "risk_probability"):
            validate_prediction(
                {"risk_probability": 1.5, "prediction": 1, "reasoning_summary": "x"}
            )

    def test_prediction_must_be_binary_integer(self):
        with self.assertRaisesRegex(LLMResponseError, "prediction"):
            validate_prediction(
                {"risk_probability": 0.5, "prediction": True, "reasoning_summary": "x"}
            )

    def test_reasoning_summary_must_not_be_empty(self):
        with self.assertRaisesRegex(LLMResponseError, "reasoning_summary"):
            validate_prediction(
                {"risk_probability": 0.5, "prediction": 0, "reasoning_summary": "  "}
            )


class APICheckTests(unittest.TestCase):
    def test_api_check_parses_and_validates_json(self):
        class FakeClient:
            def chat(self, messages, response_schema=None):
                self.messages = messages
                self.response_schema = response_schema
                return LLMResponse(
                    json.dumps(
                        {
                            "risk_probability": 0.35,
                            "prediction": 0,
                            "reasoning_summary": "Concise conclusion only.",
                        }
                    ),
                    prompt_tokens=30,
                    completion_tokens=20,
                )

        client = FakeClient()
        result = run_api_check(client)
        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertIn("/no_think", client.messages[-1]["content"])
        self.assertEqual(client.response_schema["required"], [
            "risk_probability", "prediction", "reasoning_summary"
        ])


class EnvironmentRedactionTests(unittest.TestCase):
    def test_api_key_value_is_never_returned(self):
        environment = {
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_API_KEY": "super-secret-key",
            "LLM_MODEL_NAME": "qwen3:4b",
        }
        result = redacted_llm_environment(environment)
        serialized = json.dumps(result)
        self.assertNotIn("super-secret-key", serialized)
        self.assertEqual(result["LLM_API_KEY"], "<redacted:set>")

    def test_inaccessible_command_is_reported_instead_of_crashing(self):
        with patch("scripts.check_environment.subprocess.run", side_effect=PermissionError()):
            result = _run(["ollama", "--version"])
        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "PermissionError")


if __name__ == "__main__":
    unittest.main()
