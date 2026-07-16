import unittest

from utils.llm_client import LLMResponseError


class L20PipelineDirectTests(unittest.TestCase):
    def test_extract_json_object_normalizes_only_the_first_object(self):
        from scripts.run_l20_pipeline_direct import extract_json_object

        value = extract_json_object('prefix {"prediction": 0} trailing text')

        self.assertEqual(value, {"prediction": 0})

    def test_extract_json_object_rejects_missing_json(self):
        from scripts.run_l20_pipeline_direct import extract_json_object

        with self.assertRaisesRegex(LLMResponseError, "no JSON object"):
            extract_json_object("plain text only")

    def test_validate_stage_order_requires_a_prefix_of_the_acceptance_sequence(self):
        from scripts.run_l20_pipeline_direct import validate_stage_order

        self.assertEqual(
            validate_stage_order(["single", "two-doctors", "meta"]),
            ["single", "two-doctors", "meta"],
        )
        with self.assertRaisesRegex(ValueError, "strict acceptance order"):
            validate_stage_order(["single", "meta"])

    def test_messages_with_schema_does_not_mutate_agent_messages(self):
        from scripts.run_l20_pipeline_direct import messages_with_schema

        original = [{"role": "user", "content": "fictional input"}]
        rendered = messages_with_schema(
            original,
            {"type": "object", "required": ["prediction"]},
        )

        self.assertEqual(original[0]["content"], "fictional input")
        self.assertIn("JSON Schema", rendered[0]["content"])
        self.assertIn('"prediction"', rendered[0]["content"])

    def test_summarize_calls_aggregates_all_attempts(self):
        from scripts.run_l20_pipeline_direct import summarize_calls

        summary = summarize_calls(
            [
                {"input_tokens": 10, "output_tokens": 2, "inference_seconds": 0.5},
                {"input_tokens": 12, "output_tokens": 3, "inference_seconds": 0.75},
            ]
        )

        self.assertEqual(summary["llm_call_count"], 2)
        self.assertEqual(summary["input_tokens"], 22)
        self.assertEqual(summary["output_tokens"], 5)
        self.assertEqual(summary["inference_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
