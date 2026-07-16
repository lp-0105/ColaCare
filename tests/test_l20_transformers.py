import json
from pathlib import Path
import tempfile
import unittest

from scripts.test_l20_transformers import (
    L20SmokeError,
    build_messages,
    enforce_context_limit,
    load_synthetic_fixture,
    parse_prediction_text,
    parse_with_retry,
)


ROOT = Path(__file__).resolve().parents[1]


class FixtureGateTests(unittest.TestCase):
    def test_committed_fixture_is_explicitly_fictional(self):
        fixture = load_synthetic_fixture(ROOT / "tests" / "fixtures" / "synthetic_patient.json")
        self.assertIs(fixture["synthetic"], True)
        self.assertEqual(fixture["patient_id"], "SYNTHETIC-0001")

    def test_non_synthetic_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patient.json"
            path.write_text(
                json.dumps({"synthetic": False, "patient_id": "REAL-1"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(L20SmokeError, "synthetic"):
                load_synthetic_fixture(path)


class PredictionParsingTests(unittest.TestCase):
    VALID = {
        "risk_probability": 0.35,
        "prediction": 0,
        "reasoning_summary": "Concise review of the fictional values.",
    }

    def test_json_object_inside_fence_is_parsed_and_validated(self):
        value = parse_prediction_text("```json\n" + json.dumps(self.VALID) + "\n```")
        self.assertEqual(value, self.VALID)

    def test_short_prefix_does_not_prevent_object_extraction(self):
        value = parse_prediction_text("Result:\n" + json.dumps(self.VALID))
        self.assertEqual(value["prediction"], 0)

    def test_extra_field_is_rejected_by_schema(self):
        invalid = dict(self.VALID, full_reasoning="private chain of thought")
        with self.assertRaisesRegex(L20SmokeError, "required fields"):
            parse_prediction_text(json.dumps(invalid))

    def test_parse_failure_retries_once_only(self):
        outputs = iter(
            [
                ("not json", {"input_tokens": 100, "output_tokens": 2}),
                (json.dumps(self.VALID), {"input_tokens": 110, "output_tokens": 30}),
            ]
        )
        prediction, generation, attempts = parse_with_retry(lambda attempt: next(outputs))
        self.assertEqual(prediction, self.VALID)
        self.assertEqual(generation["output_tokens"], 30)
        self.assertEqual(attempts, 2)

    def test_two_invalid_outputs_fail_without_third_call(self):
        calls = []

        def generate(attempt):
            calls.append(attempt)
            return "still invalid", {}

        with self.assertRaisesRegex(L20SmokeError, "after 2 attempts"):
            parse_with_retry(generate)
        self.assertEqual(calls, [0, 1])


class PromptAndContextTests(unittest.TestCase):
    def test_prompt_requests_short_json_without_chain_of_thought(self):
        patient = load_synthetic_fixture(ROOT / "tests" / "fixtures" / "synthetic_patient.json")
        messages = build_messages(patient, retry=False)
        serialized = json.dumps(messages)
        self.assertIn("reasoning_summary", serialized)
        self.assertIn("JSON", serialized)
        self.assertIn("Do not provide chain-of-thought", serialized)

    def test_total_context_must_not_exceed_4096(self):
        enforce_context_limit(input_tokens=3840, max_new_tokens=256, context_length=4096)
        with self.assertRaisesRegex(L20SmokeError, "4096"):
            enforce_context_limit(input_tokens=3841, max_new_tokens=256, context_length=4096)
        with self.assertRaisesRegex(L20SmokeError, "first-version limit"):
            enforce_context_limit(input_tokens=1, max_new_tokens=1, context_length=4097)

    def test_source_forces_local_bf16_deterministic_loading(self):
        source = (ROOT / "scripts" / "test_l20_transformers.py").read_text(encoding="utf-8")
        for required in (
            "local_files_only=True",
            "dtype=torch.bfloat16",
            "do_sample=False",
            "max_new_tokens",
            "apply_chat_template",
            "torch.inference_mode()",
            "traceback.print_exc()",
            "artifacts\" / \"l20_smoke",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()

