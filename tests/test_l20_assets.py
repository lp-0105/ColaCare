import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def find_test_bash():
    candidates = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "bin" / "bash.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "usr" / "bin" / "bash.exe",
    ]
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            probe = subprocess.run(
                [str(candidate), "--version"], capture_output=True, timeout=10
            )
            if probe.returncode == 0 and b"GNU bash" in probe.stdout:
                return str(candidate)
    return None


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
            "SERVED_MODEL_NAME=qwen3-4b-local",
            "LLM_BASE_URL=http://127.0.0.1:8000/v1",
            "LLM_API_KEY=EMPTY",
            "LLM_MODEL_NAME=qwen3-4b-local",
            "LLM_CONTEXT_LENGTH=4096",
            "LLM_MAX_TOKENS=256",
            "LLM_TEMPERATURE=0",
            "VLLM_MAX_NUM_SEQS=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
        ):
            self.assertIn(required, config)
        self.assertNotIn("sk-", config)

    def test_vllm_next_step_is_documented_without_install_commands_for_current_stage(self):
        doc = (ROOT / "docs" / "L20_VLLM_NEXT_STEP.md").read_text(encoding="utf-8")
        for required in (
            "本阶段不执行",
            "独立",
            "离线",
            "http://127.0.0.1:8000/v1",
            "LLM_API_KEY=EMPTY",
            "LLM_MODEL_NAME=qwen3-4b-local",
            "风险",
        ):
            self.assertIn(required, doc)

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

    def test_snapshot_packaging_script_is_pinned_and_split_with_checksums(self):
        script = (ROOT / "scripts" / "package_l20_assets.sh").read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            "git -C",
            "archive",
            "ls-tree",
            "split -b",
            "sha256sum",
            "OFFLINE_ASSET_MANIFEST.txt",
            "OFFLINE_ASSET_SHA256SUMS",
        ):
            self.assertIn(required, script)
        self.assertIn("--commit", script)
        self.assertIn("--split-size", script)

    def test_asset_verifier_is_offline_local_only_and_checks_runtime(self):
        script = (ROOT / "scripts" / "verify_l20_assets.sh").read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "sha256sum -c",
            "nvidia-smi",
            "torch.cuda.is_available",
            "torch.cuda.is_bf16_supported",
            "local_files_only=True",
            "MODEL_MANIFEST.json",
            "SHA256SUMS",
        ):
            self.assertIn(required, script)
        self.assertNotIn("curl ", script)
        self.assertNotIn("wget ", script)

    def test_transformers_runner_activates_existing_env_without_installing(self):
        script = (ROOT / "scripts" / "run_l20_transformers_smoke.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "set -euo pipefail",
            "/opt/miniconda3/envs/pytorch",
            "conda activate",
            "HF_HUB_OFFLINE=1",
            "scripts/verify_l20_assets.sh",
            "scripts/test_l20_transformers.py",
            "nvidia-smi",
            "artifacts/l20_smoke",
        ):
            self.assertIn(required, script)
        self.assertNotIn("pip install", script)
        self.assertNotIn("conda install", script)

    def test_packager_runs_on_tiny_git_commit_and_model(self):
        bash = find_test_bash()
        if bash is None:
            self.skipTest("bash is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            model = root / "Qwen3-test"
            output = root / "packages"
            repository.mkdir()
            model.mkdir()
            (repository / "README.md").write_text("tiny repository\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Offline Test"],
                check=True,
            )
            subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"], check=True
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()

            files = {
                "config.json": json.dumps({"model_type": "qwen3"}).encode(),
                "tokenizer_config.json": json.dumps({"chat_template": "{{ messages }}"}).encode(),
                "tokenizer.json": b"{}",
                "model.safetensors": b"tiny fake weight for packaging only",
            }
            for name, content in files.items():
                (model / name).write_bytes(content)
            (model / "MODEL_MANIFEST.json").write_text(
                json.dumps({"model_id": "unit/test", "revision": "abc"}), encoding="utf-8"
            )
            checksum_files = sorted(model.iterdir())
            (model / "SHA256SUMS").write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                    for path in checksum_files
                ),
                encoding="utf-8",
                newline="\n",
            )

            command = [
                bash,
                "-lc",
                'exec bash "$@"',
                "bash",
                (ROOT / "scripts" / "package_l20_assets.sh").as_posix(),
                "--model-dir",
                model.as_posix(),
                "--repo-dir",
                repository.as_posix(),
                "--commit",
                commit,
                "--output-dir",
                output.as_posix(),
                "--split-size",
                "1K",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((output / "OFFLINE_ASSET_MANIFEST.txt").is_file())
            self.assertTrue((output / "OFFLINE_ASSET_SHA256SUMS").is_file())
            self.assertEqual(len(list(output.glob("*.part-*"))), 1)
            checksum = subprocess.run(
                [bash, "-lc", f"cd '{output.as_posix()}' && sha256sum -c OFFLINE_ASSET_SHA256SUMS"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(checksum.returncode, 0, checksum.stdout + checksum.stderr)


if __name__ == "__main__":
    unittest.main()
