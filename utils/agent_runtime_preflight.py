"""Fail-closed preflight for the generated real-Agent retry4 runtime.

This module deliberately performs no model construction, patient-file reads,
or LLM calls.  It compiles and truly imports the generated runner, validates
its callback contract, and checks only the anonymous input provenance manifest.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import py_compile
import tempfile
from typing import Any, Mapping

from utils.agent_response_contract import ReasoningSummaryLengthError


EXPECTED_PLACEHOLDER_KB = "PLACEHOLDER KB — NOT PAPER RAG REPRODUCTION"
EXPECTED_VALIDATOR_SYMBOL = "validate_prediction_response"
EXPECTED_SUMMARY_TARGET = 500
EXPECTED_SUMMARY_HARD_LIMIT = 640
EXPECTED_MAX_REPAIRS = 2
PROHIBITED_AGENT_FIELDS = {
    "patientid",
    "recordid",
    "subject_id",
    "hadm_id",
    "stay_id",
    "outcome",
    "outcome_label",
    "ground_truth",
    "test_metric",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _import_runner(path: Path):
    module_name = "_colacare_retry4_preflight_" + hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for generated runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator_signature_compatible(callback: Any) -> bool:
    if not callable(callback):
        return False
    parameters = list(inspect.signature(callback).parameters.values())
    return (
        len(parameters) == 1
        and parameters[0].name == "value"
        and parameters[0].kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    )


def _repair_signature_compatible(callback: Any) -> bool:
    if not callable(callback):
        return False
    parameters = inspect.signature(callback).parameters
    return all(
        name in parameters
        for name in ("client", "messages", "response_schema", "validator")
    )


def _fixed_manifest_valid(path: Path) -> tuple[bool, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        samples = value["samples"]
    except Exception as exc:
        return False, f"cannot read anonymous input provenance manifest: {exc}"
    if not isinstance(samples, list) or len(samples) != 32:
        return False, "anonymous input provenance manifest must contain exactly 32 samples"
    expected = [f"sample_{index:06d}" for index in range(32)]
    actual = [item.get("anonymous_sample_id") for item in samples if isinstance(item, dict)]
    if actual != expected:
        return False, "anonymous sample order is not the frozen sample_000000..sample_000031 order"
    for item in samples:
        if not isinstance(item, dict):
            return False, "sample manifest entry is not an object"
        lowered_keys = {str(key).lower() for key in item}
        if lowered_keys.intersection(PROHIBITED_AGENT_FIELDS):
            return False, "sample manifest contains a prohibited label or identifier field"
        for key in ("source_input_sha256", "adapter_json_sha256", "prompt_sha256"):
            digest = item.get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                return False, f"sample manifest has an invalid {key}"
    return True, None


def run_agent_runtime_preflight(
    runner_path: Path,
    *,
    report_path: Path,
    retry3_output_dir: Path,
    retry4_output_dir: Path,
    failed_retry4_output_dir: Path | None = None,
    input_manifest_path: Path,
) -> dict[str, Any]:
    """Validate the generated runner before any Qwen/data/Agent execution."""
    runner_path = Path(runner_path)
    report_path = Path(report_path)
    failed: list[str] = []
    details: dict[str, str] = {}

    def check(name: str, condition: bool, detail: str | None = None) -> None:
        if not condition:
            failed.append(name)
            if detail:
                details[name] = detail

    runner_sha256: str | None = None
    source = ""
    module = None
    check("python_executable_readable", bool(os.path.isfile(os.sys.executable)))
    check("runner_exists", runner_path.is_file())
    if runner_path.is_file():
        runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
        source = runner_path.read_text(encoding="utf-8")

    compile_passed = False
    if runner_path.is_file():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        bytecode_path = report_path.parent / ".agent_runtime_preflight.pyc.tmp"
        try:
            py_compile.compile(str(runner_path), cfile=str(bytecode_path), doraise=True)
            compile_passed = True
        except Exception as exc:
            details["runner_py_compile_passed"] = str(exc)
        finally:
            if bytecode_path.exists():
                bytecode_path.unlink()
    check("runner_py_compile_passed", compile_passed)

    import_passed = False
    if compile_passed:
        try:
            module = _import_runner(runner_path)
            import_passed = True
        except Exception as exc:
            details["runner_true_import_passed"] = f"{type(exc).__name__}: {exc}"
    check("runner_true_import_passed", import_passed)

    doctor_callable = bool(module and callable(getattr(module, "DOCTOR_VALIDATOR", None)))
    meta_callable = bool(module and callable(getattr(module, "META_VALIDATOR", None)))
    repair_callable = bool(module and callable(getattr(module, "REPAIR_CALLBACK", None)))
    check("doctor_validator_callable", doctor_callable)
    check("meta_validator_callable", meta_callable)
    check("repair_callable", repair_callable)

    canonical = getattr(module, EXPECTED_VALIDATOR_SYMBOL, None) if module else None
    check("canonical_validator_symbol", callable(canonical))
    check(
        "doctor_validator_is_canonical",
        bool(module and getattr(module, "DOCTOR_VALIDATOR", None) is canonical),
    )
    check(
        "meta_validator_is_canonical",
        bool(module and getattr(module, "META_VALIDATOR", None) is canonical),
    )
    check("doctor_validator_signature", _validator_signature_compatible(canonical))
    check("meta_validator_signature", _validator_signature_compatible(canonical))
    check(
        "repair_callback_signature",
        _repair_signature_compatible(getattr(module, "REPAIR_CALLBACK", None) if module else None),
    )

    validator_return_mapping = False
    validator_rejects_641 = False
    if callable(canonical):
        try:
            output = canonical(
                {
                    "risk_probability": 0.5,
                    "prediction": 0,
                    "reasoning_summary": "short",
                }
            )
            validator_return_mapping = isinstance(output, dict)
            try:
                canonical(
                    {
                        "risk_probability": 0.5,
                        "prediction": 0,
                        "reasoning_summary": "中" * 641,
                    }
                )
            except ReasoningSummaryLengthError:
                validator_rejects_641 = True
        except Exception as exc:
            details["canonical_validator_behavior"] = str(exc)
    check("validator_returns_structured_mapping", validator_return_mapping)
    check("validator_rejects_641_unicode_chars", validator_rejects_641)

    summary_target = getattr(module, "SUMMARY_PROMPT_TARGET_CHARS", None) if module else None
    summary_limit = getattr(module, "SUMMARY_MAX_CHARS", None) if module else None
    max_repairs = getattr(module, "MAX_SUMMARY_REPAIRS", None) if module else None
    check("summary_target_500", summary_target == EXPECTED_SUMMARY_TARGET)
    check("summary_hard_limit_640", summary_limit == EXPECTED_SUMMARY_HARD_LIMIT)
    check("max_repairs_2", max_repairs == EXPECTED_MAX_REPAIRS)

    silent_patterns = (
        "reasoning_summary'][:",
        'reasoning_summary\"][:',
        "[:SUMMARY_MAX_CHARS]",
    )
    silent_truncation = any(pattern in source for pattern in silent_patterns)
    check("silent_truncation_disabled", not silent_truncation)

    unresolved_legacy = False
    if source:
        try:
            tree = ast.parse(source)
            unresolved_legacy = any(
                isinstance(node, ast.Name)
                and node.id == "validate_prediction"
                and isinstance(node.ctx, ast.Load)
                for node in ast.walk(tree)
            )
        except SyntaxError:
            unresolved_legacy = True
    check("no_unresolved_validate_prediction", not unresolved_legacy)
    check(
        "three_canonical_agent_call_sites",
        source.count(",validate_prediction_response,sid,") == 3,
    )

    output_builder = getattr(module, "build_output_paths", None) if module else None
    check("output_directory_builder_callable", callable(output_builder))
    retry3_resolved = retry3_output_dir.resolve()
    retry4_resolved = retry4_output_dir.resolve()
    check("retry4_output_isolated_from_retry3", retry3_resolved != retry4_resolved)
    if failed_retry4_output_dir is not None:
        check(
            "runtimefix_output_isolated_from_failed_retry4",
            failed_retry4_output_dir.resolve() != retry4_resolved,
        )
    check("retry4_output_directory_absent", not retry4_output_dir.exists())
    if callable(output_builder):
        try:
            built = output_builder(retry4_output_dir.parent)
            check("output_builder_targets_retry4", built["results"].resolve() == retry4_resolved)
        except Exception as exc:
            check("output_builder_targets_retry4", False, str(exc))

    kb_marker = getattr(module, "PLACEHOLDER_KB_CLASSIFICATION", None) if module else None
    check("placeholder_kb_marker", kb_marker == EXPECTED_PLACEHOLDER_KB)
    check(
        "placeholder_kb_marker_in_runtime_path",
        source.count(EXPECTED_PLACEHOLDER_KB) >= 2,
    )
    allowed = {
        str(value).lower()
        for value in (getattr(module, "AGENT_INPUT_ALLOWED_TOP_LEVEL_FIELDS", ()) if module else ())
    }
    check("label_fields_absent_from_agent_input_schema", not allowed.intersection(PROHIBITED_AGENT_FIELDS))
    check("id_fields_absent_from_agent_input_schema", not allowed.intersection(PROHIBITED_AGENT_FIELDS))

    manifest_valid, manifest_error = _fixed_manifest_valid(input_manifest_path)
    check("fixed_sample_configuration_valid", manifest_valid, manifest_error)

    runtime_factories_unset = bool(
        module
        and getattr(module, "RUNTIME_CLIENT_FACTORY", object()) is None
        and getattr(module, "RUNTIME_MONITOR_FACTORY", object()) is None
        and getattr(module, "RUNTIME_TORCH", object()) is None
        and getattr(module, "RUNTIME_ALLOW_SYNTHETIC_STUB", True) is False
    )
    check("production_runtime_defaults", runtime_factories_unset)

    report: dict[str, Any] = {
        "schema_version": "agent-runtime-preflight-v1",
        "passed": not failed,
        "failed_checks": failed,
        "failure_details": details,
        "python_executable": os.sys.executable,
        "runner_path": str(runner_path.resolve()),
        "runner_sha256": runner_sha256,
        "validator_symbol": EXPECTED_VALIDATOR_SYMBOL,
        "doctor_validator_callable": doctor_callable,
        "meta_validator_callable": meta_callable,
        "repair_callable": repair_callable,
        "summary_target": summary_target,
        "summary_hard_limit": summary_limit,
        "max_repairs": max_repairs,
        "silent_truncation_enabled": silent_truncation,
        "runner_py_compile_passed": compile_passed,
        "runner_true_import_passed": import_passed,
        "fixed_sample_configuration_valid": manifest_valid,
        "retry3_output_dir": str(retry3_output_dir.resolve()),
        "failed_retry4_output_dir": (
            str(failed_retry4_output_dir.resolve())
            if failed_retry4_output_dir is not None
            else None
        ),
        "retry4_output_dir": str(retry4_output_dir.resolve()),
        "input_manifest_path": str(input_manifest_path.resolve()),
        "input_sample_bodies_read": False,
        "qwen_loaded": False,
        "llm_calls_started": False,
    }
    _atomic_json(report_path, report)
    return report
