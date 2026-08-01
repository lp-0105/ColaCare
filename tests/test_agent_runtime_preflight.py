import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REAL_RETRY2_TEMPLATE = (
    ROOT / "tests" / "fixtures" / "run_real_agent_benchmark_retry2_template.py.txt"
)


def write_input_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "anonymous_sample_id": f"sample_{index:06d}",
            "source_input_sha256": hashlib.sha256(
                f"synthetic-source-{index}".encode("ascii")
            ).hexdigest(),
            "adapter_json_sha256": hashlib.sha256(
                f"synthetic-json-{index}".encode("ascii")
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                f"synthetic-prompt-{index}".encode("ascii")
            ).hexdigest(),
        }
        for index in range(32)
    ]
    path.write_text(json.dumps({"samples": samples}, sort_keys=True), encoding="utf-8")


class AgentRuntimePreflightTests(unittest.TestCase):
    def _materialize(self, root: Path) -> tuple[Path, Path]:
        from scripts.run_real_agent_benchmark_retry4 import materialize_retry4_runner

        runner = root / "generated_retry4.py"
        materialize_retry4_runner(
            REAL_RETRY2_TEMPLATE,
            runner,
            root / "generated_retry4.manifest.json",
        )
        manifest = root / "runroot" / "real_agent_inputs_32" / "input_provenance_manifest.json"
        write_input_manifest(manifest)
        return runner, manifest

    def test_preflight_imports_runner_and_passes_before_model_or_llm(self):
        from utils.agent_runtime_preflight import run_agent_runtime_preflight

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner, input_manifest = self._materialize(root)
            report_path = root / "agent_runtime_preflight_report.json"
            report = run_agent_runtime_preflight(
                runner,
                report_path=report_path,
                retry3_output_dir=root / "runroot" / "agent_outputs_v1_retry3",
                failed_retry4_output_dir=root / "runroot" / "agent_outputs_v1_retry4",
                retry4_output_dir=root / "runroot" / "agent_outputs_v1_retry4_runtimefix_v3",
                input_manifest_path=input_manifest,
            )
            self.assertTrue(report["passed"], report["failed_checks"])
            self.assertEqual(report["failed_checks"], [])
            self.assertEqual(report["validator_symbol"], "validate_prediction_response")
            self.assertTrue(report["doctor_validator_callable"])
            self.assertTrue(report["meta_validator_callable"])
            self.assertTrue(report["repair_callable"])
            self.assertEqual(report["summary_target"], 500)
            self.assertEqual(report["summary_hard_limit"], 640)
            self.assertEqual(report["max_repairs"], 2)
            self.assertFalse(report["silent_truncation_enabled"])
            self.assertFalse(report["qwen_loaded"])
            self.assertFalse(report["llm_calls_started"])
            self.assertTrue(report["fixed_sample_configuration_valid"])
            self.assertTrue(report["runner_true_import_passed"])
            self.assertTrue(report["runner_py_compile_passed"])
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")), report
            )

    def test_preflight_fails_closed_when_retry4_output_already_exists(self):
        from scripts.run_real_agent_benchmark_retry4 import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runroot = root / "runroot"
            logdir = root / "logs"
            input_manifest = runroot / "real_agent_inputs_32" / "input_provenance_manifest.json"
            write_input_manifest(input_manifest)
            (runroot / "agent_outputs_v1_retry4").mkdir(parents=True)
            (runroot / "agent_outputs_v1_retry4" / "failed-evidence.txt").write_text(
                "preserve", encoding="utf-8"
            )
            (runroot / "agent_outputs_v1_retry4_runtimefix_v3").mkdir(parents=True)
            logdir.mkdir()
            environment = {
                "RUNROOT": str(runroot),
                "LOGDIR": str(logdir),
                "RETRY4_SOURCE": str(REAL_RETRY2_TEMPLATE),
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "runpy.run_path"
            ) as run_path:
                with self.assertRaises(SystemExit) as exit_context:
                    main()
            self.assertEqual(exit_context.exception.code, 4)
            run_path.assert_not_called()
            report = json.loads(
                (logdir / "agent_runtime_preflight_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(report["passed"])
            self.assertIn("retry4_output_directory_absent", report["failed_checks"])
            self.assertFalse(report["qwen_loaded"])
            self.assertFalse(report["llm_calls_started"])
            self.assertEqual(
                (runroot / "agent_outputs_v1_retry4" / "failed-evidence.txt").read_text(
                    encoding="utf-8"
                ),
                "preserve",
            )


if __name__ == "__main__":
    unittest.main()
