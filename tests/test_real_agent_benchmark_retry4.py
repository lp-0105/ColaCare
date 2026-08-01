import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MINIMAL_RETRY2_SOURCE = """\
from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction
INPUTS=RUNROOT/'real_agent_inputs_32'; RESULTS=RUNROOT/'agent_outputs_v1_retry2'; RESULT_FILES=RESULTS/'results'
DISCUSSION_SCHEMA={'type':'object','properties':{'reasoning_summary':{'type':'string','maxLength':600}}}
def sha_bytes(value): return 'sha'
def validate_discussion(v):
 return v
def core_prediction(value): return value
def ask(client,messages,schema,validator,sid,stage,max_attempts=3):
 errors=[]
 return None,{'attempts':max_attempts,'errors':errors}
prov=json.loads((INPUTS/'input_provenance_manifest.json').read_text(encoding='utf-8'))
client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=128)
run_manifest={'generation':{'max_new_tokens':128}}
messages=[{'role':'system','content':'Return only the requested JSON.'}]
doctor=ask(client,messages,PREDICTION_SCHEMA,validate_prediction,sid,'doctor-1')
meta_initial=ask(client,messages,PREDICTION_SCHEMA,validate_prediction,sid,'meta-initial')
meta_final=ask(client,messages,PREDICTION_SCHEMA,validate_prediction,sid,'meta-final')
atomic_json(LOGDIR/'agent_smoke_8_report.json',smoke)
atomic_json(LOGDIR/'agent_benchmark_32_report.json',benchmark)
"""

REAL_RETRY2_TEMPLATE = (
    Path(__file__).parent
    / "fixtures"
    / "run_real_agent_benchmark_retry2_template.py.txt"
)
REAL_RETRY2_SHA256 = (
    "21977f9c0d993bead3f8d437e6adf0a750931433521edbd41f5b53fe54e2fda8"
)


class Retry4SourceTests(unittest.TestCase):
    def test_sha_pinned_real_retry2_template_replaces_all_legacy_agent_calls(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        source = REAL_RETRY2_TEMPLATE.read_text(encoding="utf-8")
        normalized = source.replace("\r\n", "\n").encode("utf-8")
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), REAL_RETRY2_SHA256)

        rendered = build_retry4_source(source)
        self.assertNotIn(",validate_prediction,sid,", rendered)
        self.assertEqual(
            rendered.count(",validate_prediction_response,sid,"),
            3,
            "DoctorAgent and both MetaAgent call sites must use the canonical validator",
        )

    def test_retry4_uses_isolated_output_and_leaves_retry3_untouched(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        rendered = build_retry4_source(MINIMAL_RETRY2_SOURCE)
        self.assertIn("agent_outputs_v1_retry4_runtimefix_v3", rendered)
        self.assertNotIn("agent_outputs_v1_retry3", rendered)
        self.assertNotIn("RESULTS=RUNROOT/'agent_outputs_v1_retry2'", rendered)
        self.assertIn("agent_smoke_8_report_retry4_runtimefix_v3.json", rendered)
        self.assertIn("agent_benchmark_32_report_retry4_runtimefix_v3.json", rendered)

    def test_retry4_uses_contract_repair_not_full_reinference_loop(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        rendered = build_retry4_source(MINIMAL_RETRY2_SOURCE)
        self.assertIn("request_with_summary_repair", rendered)
        self.assertNotIn("for attempt in range(1,max_attempts+1)", rendered)
        self.assertIn("max_new_tokens=256", rendered)
        self.assertIn("'max_new_tokens':256", rendered)

    def test_retry4_prompt_and_schema_contract_are_aligned(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        rendered = build_retry4_source(MINIMAL_RETRY2_SOURCE)
        self.assertIn("SUMMARY_MAX_CHARS", rendered)
        self.assertIn("SUMMARY_PROMPT_TARGET_CHARS", rendered)
        self.assertIn("500 Unicode characters", rendered)
        self.assertIn("640 Unicode characters", rendered)
        self.assertIn("Do not use Markdown", rendered)

    def test_retry4_preserves_resume_rule_for_matching_pass_results(self):
        from scripts.run_real_agent_benchmark_retry4 import retry4_resume_is_valid

        previous = {"status": "PASS", "source_input_sha256": "abc"}
        self.assertTrue(retry4_resume_is_valid(previous, "abc"))
        self.assertFalse(retry4_resume_is_valid(previous, "different"))
        self.assertFalse(retry4_resume_is_valid({"status": "FAIL", "source_input_sha256": "abc"}, "abc"))

    def test_real_generated_runner_py_compiles_and_imports_without_execution(self):
        from scripts.run_real_agent_benchmark_retry4 import (
            materialize_retry4_runner,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "generated_retry4.py"
            manifest = root / "generated_retry4_manifest.json"
            record = materialize_retry4_runner(
                REAL_RETRY2_TEMPLATE,
                runner,
                manifest,
            )
            py_compile.compile(str(runner), doraise=True)
            self.assertEqual(hashlib.sha256(runner.read_bytes()).hexdigest(), record["runner_sha256"])
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["runner_sha256"],
                record["runner_sha256"],
            )

            spec = importlib.util.spec_from_file_location("generated_retry4_test", runner)
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(spec.loader)
            with mock.patch(
                "scripts.run_l20_pipeline_direct.DirectTransformersClient"
            ) as model_client:
                spec.loader.exec_module(module)
            model_client.assert_not_called()
            self.assertTrue(callable(module.DOCTOR_VALIDATOR))
            self.assertTrue(callable(module.META_VALIDATOR))
            self.assertTrue(callable(module.REPAIR_CALLBACK))
            self.assertTrue(callable(module.build_output_paths))
            self.assertEqual(
                module.PLACEHOLDER_KB_CLASSIFICATION,
                "PLACEHOLDER KB — NOT PAPER RAG REPRODUCTION",
            )
            self.assertFalse((root / "agent_outputs_v1_retry4_runtimefix_v3").exists())


class _FakeCuda:
    def manual_seed_all(self, seed):
        self.seed = seed

    def set_device(self, device):
        self.device = device

    def reset_peak_memory_stats(self):
        return None

    def max_memory_allocated(self):
        return 0

    def max_memory_reserved(self):
        return 0


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()

    def manual_seed(self, seed):
        self.seed = seed


class _FakeMonitor:
    def __init__(self):
        self.peak_rss = 0
        self.gpu_memory_peak = 0
        self.gpu_utils = []

    def start(self):
        return None

    def stop(self):
        return None


class _StubClient:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.calls = []
        self.model_load_seconds = 0.0
        self.gpu_memory_before_load_mib = 0.0
        self.gpu_memory_after_load_mib = 0.0
        self.last_raw_response = ""
        type(self).instances.append(self)

    def chat(self, messages, response_schema=None):
        required = set(response_schema["required"])
        repair = "ORIGINAL_JSON:" in messages[-1]["content"]
        if repair:
            payload = json.loads(messages[-1]["content"].split("ORIGINAL_JSON:\n", 1)[1])
            payload["reasoning_summary"] = "Concise synthetic final summary."
        elif required == {"risk_probability", "prediction", "reasoning_summary"}:
            payload = {
                "risk_probability": 0.42,
                "prediction": 0,
                "reasoning_summary": "中" * 641,
            }
        else:
            payload = {
                "stance": "agree",
                "revised_probability": 0.42,
                "reasoning_summary": "中" * 641,
            }
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.last_raw_response = content
        self.calls.append(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "inference_seconds": 0.001,
                "messages": [dict(message) for message in messages],
            }
        )
        return SimpleNamespace(content=content, prompt_tokens=10, completion_tokens=5)


class Retry4GeneratedIntegrationTests(unittest.TestCase):
    def setUp(self):
        _StubClient.instances.clear()

    def _write_synthetic_inputs(self, runroot: Path) -> None:
        inputs = runroot / "real_agent_inputs_32"
        inputs.mkdir(parents=True)
        samples = []
        for index in range(32):
            sample_id = f"sample_{index:06d}"
            payload = {
                "anonymous_sample_id": sample_id,
                "synthetic_demo": True,
                "patient_summary": {
                    "continuous_features": {
                        "Synthetic signal": {"observed_count": 1}
                    },
                    "categorical_features": {},
                },
                "expert_predictions": {
                    "retain": {"probability": 0.41},
                    "concare": {"probability": 0.42},
                    "adacare": {"probability": 0.43},
                },
            }
            json_path = inputs / f"{sample_id}.json"
            text_path = inputs / f"{sample_id}.txt"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            text_path.write_text(
                "SYNTHETIC STUB — NOT REAL PATIENT DATA\n"
                "First-48-hour structured research summary without labels or identifiers.",
                encoding="utf-8",
            )
            samples.append(
                {
                    "anonymous_sample_id": sample_id,
                    "source_input_sha256": "0" * 64,
                    "adapter_json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
                    "prompt_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
                }
            )
        (inputs / "input_provenance_manifest.json").write_text(
            json.dumps({"samples": samples}, sort_keys=True), encoding="utf-8"
        )

    def test_real_generated_runner_executes_doctor_and_meta_with_canonical_validator(self):
        from scripts.run_real_agent_benchmark_retry4 import materialize_retry4_runner
        from utils.agent_response_contract import validate_prediction_response

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runroot = root / "runroot"
            logdir = root / "logs"
            runroot.mkdir()
            logdir.mkdir()
            self._write_synthetic_inputs(runroot)
            runner = root / "generated_retry4.py"
            materialize_retry4_runner(
                REAL_RETRY2_TEMPLATE,
                runner,
                root / "generated_manifest.json",
            )
            spec = importlib.util.spec_from_file_location("generated_retry4_smoke", runner)
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(spec.loader)
            spec.loader.exec_module(module)
            module.RUNTIME_CLIENT_FACTORY = _StubClient
            module.RUNTIME_MONITOR_FACTORY = _FakeMonitor
            module.RUNTIME_TORCH = _FakeTorch()
            module.RUNTIME_ALLOW_SYNTHETIC_STUB = True
            module.RUNTIME_INPUT_CLASSIFICATION = "SYNTHETIC STUB — NOT REAL PATIENT DATA"
            canonical = mock.Mock(wraps=validate_prediction_response)
            module.validate_prediction_response = canonical

            environment = {
                "BASE": str(root),
                "NEWREPO": str(Path(__file__).resolve().parents[1]),
                "RUNROOT": str(runroot),
                "LOGDIR": str(logdir),
                "MODEL": str(root / "not-a-model"),
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "socket.socket.connect", side_effect=AssertionError("network forbidden")
            ):
                with self.assertRaises(SystemExit) as exit_context:
                    module.deployed_main()
            smoke_path = logdir / "agent_smoke_8_report_retry4_runtimefix_v3.json"
            first_result_path = (
                runroot
                / "agent_outputs_v1_retry4_runtimefix_v3"
                / "results"
                / "sample_000000.json"
            )
            self.assertEqual(
                exit_context.exception.code,
                0,
                (
                    smoke_path.read_text(encoding="utf-8")
                    + "\nFIRST_RESULT\n"
                    + first_result_path.read_text(encoding="utf-8")
                    if smoke_path.exists() and first_result_path.exists()
                    else "missing smoke report or first result"
                ),
            )
            self.assertEqual(len(_StubClient.instances), 1)
            self.assertEqual(len(_StubClient.instances[0].calls), 408)
            all_prompts = json.dumps(
                [call["messages"] for call in _StubClient.instances[0].calls],
                ensure_ascii=False,
            ).lower()
            for prohibited in (
                "patientid",
                "recordid",
                "subject_id",
                "hadm_id",
                "stay_id",
                "outcome_label",
                "ground_truth",
                "test_metric",
            ):
                self.assertNotIn(prohibited, all_prompts)
            self.assertEqual(canonical.call_count, 272)
            self.assertEqual(len(list((runroot / "agent_outputs_v1_retry4_runtimefix_v3" / "results").glob("sample_*.json"))), 32)
            benchmark = json.loads(
                (logdir / "agent_benchmark_32_report_retry4_runtimefix_v3.json").read_text(encoding="utf-8")
            )
            self.assertEqual(benchmark["success"], 32)
            self.assertEqual(benchmark["failed"], 0)
            self.assertTrue(benchmark["privacy_all_pass"])
            result = json.loads(
                (runroot / "agent_outputs_v1_retry4_runtimefix_v3" / "results" / "sample_000000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["kb_classification"],
                "PLACEHOLDER KB — NOT PAPER RAG REPRODUCTION",
            )
            self.assertEqual(len(result["doctor_results"]), 2)
            self.assertIn("initial_meta", result)
            self.assertIn("final_consensus", result)
            self.assertEqual(
                result["doctor_results"][0]["reasoning_summary"],
                "Concise synthetic final summary.",
            )
            raw_root = runroot / "agent_outputs_v1_retry4_runtimefix_v3" / "raw_responses"
            raw_responses = list(raw_root.glob("sample_*/*.initial.raw.txt"))
            raw_hashes = list(raw_root.glob("sample_*/*.initial.raw.txt.sha256"))
            self.assertEqual(len(raw_responses), 192)
            self.assertEqual(len(raw_hashes), 192)
            first_raw = raw_responses[0]
            self.assertEqual(
                Path(str(first_raw) + ".sha256").read_text(encoding="ascii").strip(),
                hashlib.sha256(first_raw.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["input_classification"],
                "SYNTHETIC STUB — NOT REAL PATIENT DATA",
            )
            serialized = json.dumps(result, ensure_ascii=False).lower()
            for prohibited in (
                "patientid",
                "recordid",
                "subject_id",
                "hadm_id",
                "stay_id",
                "outcome_label",
                "ground_truth",
                "test_metric",
            ):
                self.assertNotIn(prohibited, serialized)


if __name__ == "__main__":
    unittest.main()
