"""Run the real 8/32 Agent gate with summary-only JSON repair.

This launcher intentionally reuses the already-validated retry2 runtime source
from ``RUNROOT``. It writes only to a new runtime-fix directory and new retry4
report names. The script contains no data, identifiers, checkpoints, or
machine-specific paths; callers provide paths through environment variables.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def retry4_resume_is_valid(previous: dict, source_input_sha256: str) -> bool:
    return (
        previous.get("status") == "PASS"
        and previous.get("source_input_sha256") == source_input_sha256
    )


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"retry2 source contract changed for {old!r}: expected 1, found {count}"
        )
    return source.replace(old, new, 1)


def _replace_exact_count(
    source: str, old: str, new: str, *, expected_count: int, name: str
) -> str:
    count = source.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"retry2 source contract changed for {name}: "
            f"expected {expected_count}, found {count}"
        )
    return source.replace(old, new)


def _replace_block(source: str, pattern: str, replacement: str, name: str) -> str:
    rendered, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"retry2 source contract changed: missing {name} block")
    return rendered


_ASK_REPLACEMENT = '''def ask(client,messages,schema,validator,sid,stage,max_attempts=3):
 # Contract: target at most 500 Unicode characters; hard limit 640 Unicode characters; Do not use Markdown.
 before=len(client.calls); started=time.perf_counter()
 contract_validator=(validate_prediction_response if set(schema.get('required',[]))=={'risk_probability','prediction','reasoning_summary'} else validate_discussion_response)
 raw_path=RESULTS/'raw_responses'/sid/f'{stage}.initial.raw.txt'
 try:
  outcome=request_with_summary_repair(client,messages,response_schema=schema,validator=contract_validator,original_response_path=raw_path)
  new=client.calls[before:]
  records=[]
  for offset,item in enumerate(new):
   kind='INITIAL_REASONING' if offset==0 else 'SUMMARY_REPAIR'
   records.append({**item,'sample_id':sid,'stage':stage,'attempt':offset+1,'call_kind':kind,'status':'PASS'})
  return outcome.payload,{'attempts':1+outcome.repair_attempts,'repair_attempts':outcome.repair_attempts,'errors':outcome.repair_errors,'wall_seconds':round(time.perf_counter()-started,4),'calls':records,'usage':{'prompt_tokens':outcome.prompt_tokens,'completion_tokens':outcome.completion_tokens},'original_response_sha256':outcome.original_response_sha256,'original_summary_length':outcome.original_summary_length}
 except Exception as exc:
  new=client.calls[before:]
  records=[]
  for offset,item in enumerate(new):
   kind='INITIAL_REASONING' if offset==0 else 'SUMMARY_REPAIR'
   records.append({**item,'sample_id':sid,'stage':stage,'attempt':offset+1,'call_kind':kind,'status':'PARSE_OR_VALIDATION_FAIL'})
  return None,{'attempts':len(new),'repair_attempts':max(0,len(new)-1),'errors':[{'attempt':len(new),'type':type(exc).__name__,'message':str(exc)[:500]}],'wall_seconds':round(time.perf_counter()-started,4),'calls':records,'usage':{'prompt_tokens':sum(int(r.get('input_tokens',0)) for r in records),'completion_tokens':sum(int(r.get('output_tokens',0)) for r in records)}}
'''


def build_retry4_source(source: str) -> str:
    """Apply guarded, auditable transformations to the validated retry2 runner."""
    source = _replace_once(
        source,
        "from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction",
        "from utils.agent_response_contract import (DISCUSSION_RESPONSE_SCHEMA, "
        "PREDICTION_RESPONSE_SCHEMA as PREDICTION_SCHEMA, MAX_SUMMARY_REPAIRS, "
        "SUMMARY_MAX_CHARS, "
        "SUMMARY_PROMPT_TARGET_CHARS, request_with_summary_repair, "
        "validate_discussion_response, validate_prediction_response)",
    )
    source = _replace_once(
        source,
        "RESULTS=RUNROOT/'agent_outputs_v1_retry2'",
        "RESULTS=RUNROOT/'agent_outputs_v1_retry4_runtimefix_v3'",
    )
    source = _replace_block(
        source,
        r"DISCUSSION_SCHEMA=.*?\ndef sha_bytes",
        "DISCUSSION_SCHEMA=DISCUSSION_RESPONSE_SCHEMA\ndef sha_bytes",
        "discussion schema",
    )
    source = _replace_block(
        source,
        r"def validate_discussion\(v\):.*?\ndef core_prediction",
        "def validate_discussion(v): return validate_discussion_response(v)\n"
        "def core_prediction",
        "discussion validator",
    )
    source = _replace_block(
        source,
        r"def ask\(client,messages,schema,validator,sid,stage,max_attempts=3\):.*?\nprov=",
        _ASK_REPLACEMENT + "prov=",
        "ask",
    )
    source = _replace_exact_count(
        source,
        ",validate_prediction,sid,",
        ",validate_prediction_response,sid,",
        expected_count=3,
        name="DoctorAgent and MetaAgent prediction-validator call sites",
    )
    source = _replace_once(
        source,
        "client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=128)",
        "client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=256)",
    )
    source = _replace_once(source, "'max_new_tokens':128", "'max_new_tokens':256")
    source = _replace_once(
        source,
        "LOGDIR/'agent_smoke_8_report.json'",
        "LOGDIR/'agent_smoke_8_report_retry4_runtimefix_v3.json'",
    )
    source = _replace_once(
        source,
        "LOGDIR/'agent_benchmark_32_report.json'",
        "LOGDIR/'agent_benchmark_32_report_retry4_runtimefix_v3.json'",
    )
    return source


PLACEHOLDER_KB_CLASSIFICATION = "PLACEHOLDER KB — NOT PAPER RAG REPRODUCTION"
RETRY4_OUTPUT_DIRECTORY_NAME = "agent_outputs_v1_retry4_runtimefix_v3"
EXPECTED_RETRY2_TEMPLATE_SHA256 = (
    "21977f9c0d993bead3f8d437e6adf0a750931433521edbd41f5b53fe54e2fda8"
)


def build_import_safe_retry4_source(source: str) -> str:
    """Render the deployment runner so importing it has no execution side effects.

    The complete validated retry2 body remains intact inside ``deployed_main``.
    Dependency-injection globals exist only so the exact generated call graph can
    be exercised with synthetic stubs before deployment; production defaults are
    the original torch, monitor, and DirectTransformersClient implementations.
    """
    rendered = build_retry4_source(source)
    marker = "BASE=Path(os.environ['BASE'])"
    if rendered.count(marker) != 1:
        raise RuntimeError("retry2 source contract changed: runtime body marker")
    imports, body = rendered.split(marker, 1)
    imports = imports.replace("import torch\n", "import torch as _retry4_torch\n")
    body = marker + body
    body = _replace_once(
        body,
        "monitor=ResourceMonitor(); monitor.start()",
        "monitor=(RUNTIME_MONITOR_FACTORY or ResourceMonitor)(); monitor.start()",
    )
    body = _replace_once(
        body,
        "client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=256)",
        "client=(RUNTIME_CLIENT_FACTORY or DirectTransformersClient)(MODEL,device='cuda:0',context_length=4096,max_new_tokens=256)",
    )
    body = _replace_once(
        body,
        "assert sha_file(jp)==entry['adapter_json_sha256'] and sha_file(tp)==entry['prompt_sha256'] and patient['anonymous_sample_id']==sid and patient.get('synthetic_demo') is False",
        "assert sha_file(jp)==entry['adapter_json_sha256'] and sha_file(tp)==entry['prompt_sha256'] and patient['anonymous_sample_id']==sid and patient.get('synthetic_demo') is RUNTIME_ALLOW_SYNTHETIC_STUB",
    )
    body = _replace_once(
        body,
        "'schema_version':'real-expert-agent-smoke-v1',",
        "'schema_version':('synthetic-agent-runtime-stub-v1' if RUNTIME_ALLOW_SYNTHETIC_STUB else 'real-expert-agent-smoke-v1'),'input_classification':RUNTIME_INPUT_CLASSIFICATION,",
    )
    body = _replace_once(
        body,
        "'test_metrics_provided':False",
        "'evaluation_metrics_provided':False",
    )
    indented = "\n".join("    " + line if line else "" for line in body.splitlines())
    module_contract = f'''\nDOCTOR_VALIDATOR = validate_prediction_response
META_VALIDATOR = validate_prediction_response
REPAIR_CALLBACK = request_with_summary_repair
PLACEHOLDER_KB_CLASSIFICATION = {PLACEHOLDER_KB_CLASSIFICATION!r}
OUTPUT_DIRECTORY_NAME = {RETRY4_OUTPUT_DIRECTORY_NAME!r}
AGENT_INPUT_ALLOWED_TOP_LEVEL_FIELDS = (
    "schema_version", "anonymous_sample_id", "split", "observation_window_hours",
    "patient_summary", "expert_predictions", "expert_evidence",
    "agreement_summary", "provenance", "synthetic_demo",
)
PROHIBITED_AGENT_FIELDS = (
    "PatientID", "RecordID", "subject_id", "hadm_id", "stay_id",
    "outcome", "outcome_label", "ground_truth", "test_metric",
)
RUNTIME_CLIENT_FACTORY = None
RUNTIME_MONITOR_FACTORY = None
RUNTIME_TORCH = None
RUNTIME_ALLOW_SYNTHETIC_STUB = False
RUNTIME_INPUT_CLASSIFICATION = "ANONYMOUS FORMAL_V2 EXPERT INPUT"


def build_output_paths(runroot):
    root = Path(runroot)
    results = root / OUTPUT_DIRECTORY_NAME
    return {{
        "results": results,
        "result_files": results / "results",
        "repeat_checks": results / "repeat_checks",
    }}


def deployed_main():
    torch = RUNTIME_TORCH or _retry4_torch
{indented}


if __name__ == "__main__":
    deployed_main()
'''
    return imports.rstrip() + "\n" + module_contract


def _write_new_or_same(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different generated file: {path}")
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def materialize_retry4_runner(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Generate the exact deployment runtime and an auditable manifest."""
    source = source_path.read_text(encoding="utf-8")
    normalized_source = source.replace("\r\n", "\n")
    source_sha256 = hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()
    if source_sha256 != EXPECTED_RETRY2_TEMPLATE_SHA256:
        raise RuntimeError(
            "validated retry2 source SHA256 mismatch: "
            f"expected {EXPECTED_RETRY2_TEMPLATE_SHA256}, found {source_sha256}"
        )
    runner = build_import_safe_retry4_source(normalized_source)
    _write_new_or_same(output_path, runner)
    runner_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("runner_sha256") != runner_sha256
            or existing.get("template_sha256") != source_sha256
        ):
            raise FileExistsError(
                f"refusing to overwrite incompatible generation manifest: {manifest_path}"
            )
        return existing
    record: dict[str, object] = {
        "schema_version": "retry4-generated-runner-v1",
        "generator_entry": "scripts.run_real_agent_benchmark_retry4:materialize_retry4_runner",
        "template_path": str(source_path.resolve()),
        "template_sha256": source_sha256,
        "runner_path": str(output_path.resolve()),
        "runner_sha256": runner_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_configuration": {
            "import_safe": True,
            "production_defaults_unchanged": True,
            "synthetic_stub_injection_disabled_by_default": True,
        },
        "deployment_structure": [
            "retry configuration",
            "DoctorAgent parse and canonical validation",
            "summary-only repair and field invariance",
            "MetaAgent parse and canonical validation",
            "isolated retry4 output and resume paths",
            "placeholder KB, privacy, and label gates",
        ],
    }
    _write_new_or_same(
        manifest_path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return record


def main() -> None:
    runroot = Path(os.environ["RUNROOT"])
    logdir = Path(os.environ["LOGDIR"])
    source_path = Path(
        os.environ.get(
            "RETRY4_SOURCE",
            str(runroot / "run_real_agent_benchmark_retry2.py"),
        )
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"validated retry2 source is missing: {source_path}")
    generated_path = Path(
        os.environ.get(
            "RETRY4_GENERATED_PATH",
            str(logdir / "generated" / "run_real_agent_benchmark_retry4.generated.py"),
        )
    )
    generation_manifest = generated_path.with_suffix(".manifest.json")
    materialize_retry4_runner(source_path, generated_path, generation_manifest)

    from utils.agent_runtime_preflight import run_agent_runtime_preflight

    preflight = run_agent_runtime_preflight(
        generated_path,
        report_path=logdir / "agent_runtime_preflight_report.json",
        retry3_output_dir=runroot / "agent_outputs_v1_retry3",
        failed_retry4_output_dir=runroot / "agent_outputs_v1_retry4",
        retry4_output_dir=runroot / RETRY4_OUTPUT_DIRECTORY_NAME,
        input_manifest_path=(
            runroot / "real_agent_inputs_32" / "input_provenance_manifest.json"
        ),
    )
    if not preflight["passed"]:
        raise SystemExit(4)
    if os.environ.get("RETRY4_PREFLIGHT_ONLY") == "1":
        return
    import runpy

    runpy.run_path(str(generated_path), run_name="__main__")


if __name__ == "__main__":
    main()
