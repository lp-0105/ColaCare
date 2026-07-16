import json
import socket
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from utils.llm_client import (
    LLMClient,
    LLMConnectionError,
    LLMConfigurationError,
    LLMHTTPError,
    LLMResponseError,
    LLMSettings,
    LLMTimeoutError,
    parse_chat_response,
    parse_json_object,
)


class LLMSettingsTests(unittest.TestCase):
    def test_defaults_target_local_ollama(self):
        settings = LLMSettings.from_env({})
        self.assertEqual(settings.base_url, "http://localhost:11434/v1")
        self.assertEqual(settings.api_key, "ollama")
        self.assertEqual(settings.model_name, "qwen3:4b")
        self.assertEqual(settings.max_tokens, 256)
        self.assertEqual(settings.context_length, 4096)
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.reasoning_effort, "none")

    def test_base_url_is_normalized_to_v1(self):
        settings = LLMSettings.from_env({"LLM_BASE_URL": "http://127.0.0.1:11434/"})
        self.assertEqual(settings.base_url, "http://127.0.0.1:11434/v1")

    def test_existing_v1_suffix_is_not_duplicated(self):
        settings = LLMSettings.from_env({"LLM_BASE_URL": "http://localhost:8000/v1/"})
        self.assertEqual(settings.base_url, "http://localhost:8000/v1")

    def test_vllm_switch_uses_only_openai_compatible_endpoint_key_and_model(self):
        settings = LLMSettings.from_env(
            {
                "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
                "LLM_API_KEY": "EMPTY",
                "LLM_MODEL_NAME": "qwen3-4b-local",
            }
        )
        self.assertEqual(settings.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(settings.api_key, "EMPTY")
        self.assertEqual(settings.model_name, "qwen3-4b-local")
        self.assertEqual(settings.context_length, 4096)
        self.assertEqual(settings.max_tokens, 256)
        self.assertEqual(settings.temperature, 0.0)

    def test_invalid_numeric_value_has_variable_name(self):
        with self.assertRaisesRegex(LLMConfigurationError, "LLM_MAX_TOKENS"):
            LLMSettings.from_env({"LLM_MAX_TOKENS": "many"})

    def test_max_tokens_must_fit_context(self):
        with self.assertRaisesRegex(LLMConfigurationError, "LLM_MAX_TOKENS"):
            LLMSettings.from_env({"LLM_MAX_TOKENS": "5000", "LLM_CONTEXT_LENGTH": "4096"})


class ResponseParsingTests(unittest.TestCase):
    def test_parse_chat_response_returns_content_and_usage(self):
        envelope = {
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        parsed = parse_chat_response(envelope)
        self.assertEqual(parsed.content, '{"ok": true}')
        self.assertEqual(parsed.prompt_tokens, 12)
        self.assertEqual(parsed.completion_tokens, 4)

    def test_empty_content_is_rejected(self):
        with self.assertRaisesRegex(LLMResponseError, "empty"):
            parse_chat_response({"choices": [{"message": {"content": "  "}}]})

    def test_malformed_envelope_is_rejected(self):
        with self.assertRaisesRegex(LLMResponseError, "choices"):
            parse_chat_response({"result": "missing choices"})

    def test_plain_json_object_is_parsed(self):
        self.assertEqual(parse_json_object('{"prediction": 1}'), {"prediction": 1})

    def test_fenced_json_object_is_parsed(self):
        value = parse_json_object('```json\n{"prediction": 0}\n```')
        self.assertEqual(value, {"prediction": 0})

    def test_non_object_json_is_rejected(self):
        with self.assertRaisesRegex(LLMResponseError, "object"):
            parse_json_object(json.dumps([1, 2, 3]))

    def test_invalid_json_is_rejected_without_echoing_payload(self):
        secret_text = "not-json-PATIENT-SECRET"
        with self.assertRaises(LLMResponseError) as caught:
            parse_json_object(secret_text)
        self.assertNotIn("PATIENT-SECRET", str(caught.exception))


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = LLMSettings.from_env({})

    def test_chat_builds_openai_compatible_schema_request(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return {
                "choices": [{"message": {"content": '{"prediction": 0}'}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }

        schema = {
            "type": "object",
            "properties": {"prediction": {"type": "integer"}},
            "required": ["prediction"],
        }
        response = LLMClient(self.settings, transport=transport).chat(
            [{"role": "user", "content": "Return JSON"}], response_schema=schema
        )
        self.assertEqual(captured["url"], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer ollama")
        self.assertEqual(captured["payload"]["model"], "qwen3:4b")
        self.assertEqual(captured["payload"]["max_tokens"], 256)
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["reasoning_effort"], "none")
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(
            captured["payload"]["response_format"]["json_schema"]["schema"], schema
        )
        self.assertEqual(response.prompt_tokens, 8)

    def test_connection_refused_has_actionable_error(self):
        client = LLMClient(self.settings)
        with patch("utils.llm_client.urlopen", side_effect=URLError("refused")):
            with self.assertRaisesRegex(LLMConnectionError, "service"):
                client.chat([{"role": "user", "content": "hello"}])

    def test_socket_timeout_has_actionable_error(self):
        client = LLMClient(self.settings)
        with patch("utils.llm_client.urlopen", side_effect=socket.timeout("slow")):
            with self.assertRaisesRegex(LLMTimeoutError, "timed out"):
                client.chat([{"role": "user", "content": "hello"}])

    def test_http_error_does_not_echo_response_body(self):
        error = HTTPError(
            "http://localhost:11434/v1/chat/completions",
            500,
            "failure-PATIENT-SECRET",
            None,
            None,
        )
        client = LLMClient(self.settings)
        with patch("utils.llm_client.urlopen", side_effect=error):
            with self.assertRaises(LLMHTTPError) as caught:
                client.chat([{"role": "user", "content": "hello"}])
        self.assertIn("500", str(caught.exception))
        self.assertNotIn("PATIENT-SECRET", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
