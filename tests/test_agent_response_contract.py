import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from utils.llm_client import LLMResponse


def prediction(summary: str, *, probability: float = 0.35) -> dict:
    return {
        "risk_probability": probability,
        "prediction": 0,
        "reasoning_summary": summary,
    }


class QueueClient:
    def __init__(self, raw_responses: list[str]):
        self.raw_responses = list(raw_responses)
        self.calls = []
        self.last_raw_response = None

    def chat(self, messages, response_schema=None):
        raw = self.raw_responses.pop(0)
        self.last_raw_response = raw
        self.calls.append({"messages": messages, "schema": response_schema})
        return LLMResponse(raw, prompt_tokens=10, completion_tokens=5)


class RawOverrideClient(QueueClient):
    def __init__(self, content: str, raw: str):
        super().__init__([content])
        self.raw_override = raw

    def chat(self, messages, response_schema=None):
        response = super().chat(messages, response_schema=response_schema)
        self.last_raw_response = self.raw_override
        return response


class SummaryBoundaryTests(unittest.TestCase):
    def test_public_prediction_and_discussion_schemas_share_640_limit(self):
        from utils.prediction_schema import PREDICTION_SCHEMA
        from utils.smoke_agents import DISCUSSION_SCHEMA

        self.assertEqual(
            PREDICTION_SCHEMA["properties"]["reasoning_summary"]["maxLength"],
            640,
        )
        self.assertEqual(
            DISCUSSION_SCHEMA["properties"]["reasoning_summary"]["maxLength"],
            640,
        )

    def test_639_and_640_unicode_characters_pass(self):
        from utils.agent_response_contract import validate_prediction_response

        self.assertEqual(
            len(validate_prediction_response(prediction("中" * 639))["reasoning_summary"]),
            639,
        )
        self.assertEqual(
            len(validate_prediction_response(prediction("中" * 640))["reasoning_summary"]),
            640,
        )

    def test_641_unicode_characters_are_rejected(self):
        from utils.agent_response_contract import ReasoningSummaryLengthError
        from utils.agent_response_contract import validate_prediction_response

        with self.assertRaises(ReasoningSummaryLengthError) as caught:
            validate_prediction_response(prediction("中" * 641))
        self.assertEqual(caught.exception.actual_length, 641)
        self.assertEqual(caught.exception.maximum_length, 640)

    def test_unicode_limit_counts_characters_not_utf8_bytes(self):
        from utils.agent_response_contract import validate_prediction_response

        summary = "风险可控。" * 100
        self.assertGreater(len(summary.encode("utf-8")), 640)
        self.assertLessEqual(len(summary), 640)
        self.assertEqual(validate_prediction_response(prediction(summary))["reasoning_summary"], summary)

    def test_markdown_is_rejected(self):
        from utils.agent_response_contract import ResponseContractError
        from utils.agent_response_contract import validate_prediction_response

        with self.assertRaisesRegex(ResponseContractError, "Markdown"):
            validate_prediction_response(prediction("- bullet"))


class RepairPromptTests(unittest.TestCase):
    def test_repair_prompt_contains_actual_length_limit_and_scope(self):
        from utils.agent_response_contract import build_summary_repair_messages

        messages = build_summary_repair_messages(prediction("x" * 641))
        rendered = "\n".join(item["content"] for item in messages)
        self.assertIn("641", rendered)
        self.assertIn("640", rendered)
        self.assertIn("500", rendered)
        self.assertIn("only reasoning_summary", rendered)
        self.assertIn("complete valid JSON", rendered)

    def test_repair_prompt_has_no_label_or_patient_identifiers(self):
        from utils.agent_response_contract import build_summary_repair_messages

        rendered = json.dumps(
            build_summary_repair_messages(prediction("x" * 641)),
            ensure_ascii=False,
        ).lower()
        for token in ("patientid", "subject_id", "hadm_id", "stay_id", "outcome_label"):
            self.assertNotIn(token, rendered)

    def test_summary_only_comparison_rejects_evidence_or_conclusion_changes(self):
        from utils.agent_response_contract import (
            ResponseContractError,
            validate_summary_only_change,
        )

        original = {
            "reasoning_summary": "x" * 641,
            "probability": 0.4,
            "evidence": ["synthetic-evidence-a"],
            "conclusion": "lower research-model estimate",
        }
        for protected_field, changed_value in (
            ("evidence", ["synthetic-evidence-b"]),
            ("conclusion", "changed conclusion"),
        ):
            repaired = dict(original)
            repaired["reasoning_summary"] = "short"
            repaired[protected_field] = changed_value
            with self.subTest(field=protected_field), self.assertRaisesRegex(
                ResponseContractError, protected_field
            ):
                validate_summary_only_change(original, repaired)


class SummaryRepairFlowTests(unittest.TestCase):
    def test_label_or_patient_identifier_in_initial_prompt_is_rejected_before_call(self):
        from utils.agent_response_contract import ResponseContractError
        from utils.agent_response_contract import request_prediction_with_summary_repair

        for token in ("outcome_label", "PatientID", "subject_id", "hadm_id", "stay_id"):
            client = QueueClient([json.dumps(prediction("short"))])
            with self.subTest(token=token), self.assertRaisesRegex(
                ResponseContractError, "prohibited"
            ):
                request_prediction_with_summary_repair(
                    client,
                    [{"role": "user", "content": f"unsafe {token}"}],
                )
            self.assertEqual(client.calls, [])

    def test_patient_identifier_token_in_output_is_rejected(self):
        from utils.agent_response_contract import ResponseContractError
        from utils.agent_response_contract import request_prediction_with_summary_repair

        client = QueueClient(
            [json.dumps(prediction("The subject_id field supports this result."))]
        )
        with self.assertRaisesRegex(ResponseContractError, "prohibited"):
            request_prediction_with_summary_repair(
                client, [{"role": "user", "content": "safe"}]
            )

    def test_patient_identifier_token_in_raw_prefix_is_rejected(self):
        from utils.agent_response_contract import ResponseContractError
        from utils.agent_response_contract import request_prediction_with_summary_repair

        content = json.dumps(prediction("short"))
        client = RawOverrideClient(content, "subject_id leaked prefix\n" + content)
        with self.assertRaisesRegex(ResponseContractError, "prohibited"):
            request_prediction_with_summary_repair(
                client, [{"role": "user", "content": "safe"}]
            )

    def test_valid_json_is_parsed_without_repair(self):
        from utils.agent_response_contract import request_prediction_with_summary_repair

        original_messages = [{"role": "user", "content": "safe"}]
        client = QueueClient([json.dumps(prediction("short conclusion"))])
        result = request_prediction_with_summary_repair(client, original_messages)
        self.assertEqual(result.payload["risk_probability"], 0.35)
        self.assertEqual(result.repair_attempts, 0)
        self.assertEqual(len(client.calls), 1)
        initial_prompt = "\n".join(
            message["content"] for message in client.calls[0]["messages"]
        )
        self.assertIn("500 Unicode characters", initial_prompt)
        self.assertIn("640 Unicode characters", initial_prompt)
        self.assertIn("Do not use Markdown", initial_prompt)
        self.assertEqual(original_messages, [{"role": "user", "content": "safe"}])

    def test_overlong_summary_uses_dedicated_repair_and_preserves_other_fields(self):
        from utils.agent_response_contract import request_prediction_with_summary_repair

        original = json.dumps(prediction("x" * 641, probability=0.37), ensure_ascii=False)
        repaired = json.dumps(prediction("short final summary", probability=0.37), ensure_ascii=False)
        client = QueueClient([original, repaired])
        result = request_prediction_with_summary_repair(client, [{"role": "user", "content": "safe"}])

        self.assertEqual(result.payload["reasoning_summary"], "short final summary")
        self.assertEqual(result.payload["risk_probability"], 0.37)
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(len(client.calls), 2)
        repair_text = "\n".join(item["content"] for item in client.calls[1]["messages"])
        self.assertNotIn("safe", repair_text)
        self.assertIn("641", repair_text)

    def test_repair_that_changes_probability_is_rejected_then_second_repair_can_pass(self):
        from utils.agent_response_contract import request_prediction_with_summary_repair

        original = json.dumps(prediction("x" * 641, probability=0.37))
        changed = json.dumps(prediction("short", probability=0.91))
        valid = json.dumps(prediction("final short", probability=0.37))
        client = QueueClient([original, changed, valid])
        result = request_prediction_with_summary_repair(client, [{"role": "user", "content": "safe"}])

        self.assertEqual(result.payload["risk_probability"], 0.37)
        self.assertEqual(result.repair_attempts, 2)
        self.assertEqual(result.repair_errors[0]["code"], "non_summary_fields_changed")

    def test_malformed_first_repair_uses_only_the_second_repair_slot(self):
        from utils.agent_response_contract import request_prediction_with_summary_repair

        original = json.dumps(prediction("x" * 641))
        valid = json.dumps(prediction("final short"))
        client = QueueClient([original, "not JSON", valid])
        result = request_prediction_with_summary_repair(
            client, [{"role": "user", "content": "safe"}]
        )

        self.assertEqual(result.repair_attempts, 2)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(result.repair_errors[0]["code"], "repair_contract_error")

    def test_two_failed_repairs_stop_without_silent_truncation(self):
        from utils.agent_response_contract import SummaryRepairExhaustedError
        from utils.agent_response_contract import request_prediction_with_summary_repair

        raw = json.dumps(prediction("中" * 641), ensure_ascii=False)
        client = QueueClient([raw, raw, raw])
        with self.assertRaises(SummaryRepairExhaustedError) as caught:
            request_prediction_with_summary_repair(client, [{"role": "user", "content": "safe"}])

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(caught.exception.original_summary_length, 641)
        self.assertNotIn("中" * 640, str(caught.exception))

    def test_original_failed_response_and_sha256_are_preserved(self):
        from utils.agent_response_contract import request_prediction_with_summary_repair

        original = json.dumps(prediction("x" * 641), ensure_ascii=False)
        repaired = json.dumps(prediction("short"), ensure_ascii=False)
        client = QueueClient([original, repaired])
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "original_response.json"
            result = request_prediction_with_summary_repair(
                client,
                [{"role": "user", "content": "safe"}],
                original_response_path=raw_path,
            )
            self.assertEqual(raw_path.read_text(encoding="utf-8"), original)
            self.assertEqual(
                result.original_response_sha256,
                hashlib.sha256(original.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                (Path(str(raw_path) + ".sha256")).read_text(encoding="ascii").strip(),
                result.original_response_sha256,
            )


if __name__ == "__main__":
    unittest.main()
