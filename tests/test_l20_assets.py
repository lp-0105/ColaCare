from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class L20AssetTests(unittest.TestCase):
    def test_vllm_start_script_is_offline_and_bounded(self):
        script = (ROOT / "scripts" / "start_vllm_server.sh").read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "vllm serve",
            "--served-model-name",
            "--max-model-len",
            "--max-num-seqs",
            "--gpu-memory-utilization",
            "MODEL_DIR",
        ):
            self.assertIn(required, script)

    def test_l20_smoke_script_runs_same_three_acceptance_checks(self):
        script = (ROOT / "scripts" / "run_l20_smoke_test.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", script)
        self.assertIn("scripts/check_environment.py", script)
        self.assertIn("scripts/test_llm_api.py", script)
        self.assertIn("scripts/run_synthetic_smoke.py --stage single", script)

    def test_example_config_has_no_live_secret_and_uses_single_sequence(self):
        config = (ROOT / "configs" / "l20.env.example").read_text(encoding="utf-8")
        for required in (
            "MODEL_DIR=",
            "LLM_BASE_URL=",
            "LLM_API_KEY=offline-local",
            "LLM_MODEL_NAME=",
            "LLM_CONTEXT_LENGTH=4096",
            "LLM_MAX_TOKENS=256",
            "LLM_TEMPERATURE=0",
            "VLLM_MAX_NUM_SEQS=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
        ):
            self.assertIn(required, config)
        self.assertNotIn("sk-", config)

    def test_deployment_doc_covers_required_offline_workflow(self):
        doc = (ROOT / "docs" / "L20_OFFLINE_DEPLOYMENT.md").read_text(encoding="utf-8")
        for required in (
            "预下载",
            "SHA256",
            "上传",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "OpenAI 兼容",
            "上下文与并发",
            "逐步扩量",
            "断点续跑",
            ".gitignore",
        ):
            self.assertIn(required, doc)


if __name__ == "__main__":
    unittest.main()
