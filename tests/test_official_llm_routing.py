from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LLM_CALL_FILES = [
    ROOT / "utils" / "framework.py",
    ROOT / "utils" / "llm.py",
    ROOT / "baselines" / "llm.py",
]


class OfficialLLMRoutingTests(unittest.TestCase):
    def test_official_llm_paths_do_not_construct_openai_sdk_clients(self):
        for path in LLM_CALL_FILES:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("from openai import OpenAI", source, path.as_posix())
            self.assertNotIn("chat.completions.create", source, path.as_posix())

    def test_common_agent_uses_environment_adapter(self):
        source = (ROOT / "utils" / "framework.py").read_text(encoding="utf-8")
        self.assertIn("LLMClient.from_env()", source)

    def test_env_example_lists_every_required_llm_variable(self):
        source = (ROOT / ".env.example").read_text(encoding="utf-8")
        for name in (
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "LLM_MODEL_NAME",
            "LLM_MAX_TOKENS",
            "LLM_CONTEXT_LENGTH",
            "LLM_TEMPERATURE",
        ):
            self.assertIn(name + "=", source)


if __name__ == "__main__":
    unittest.main()
