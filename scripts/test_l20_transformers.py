"""Run one offline BF16 Transformers prediction on a fictional ColaCare fixture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.llm_client import LLMResponseError
from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DEFAULT_FIXTURE = FIXTURE_ROOT / "synthetic_patient.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "l20_smoke"
FIRST_VERSION_CONTEXT_LIMIT = 4096
DEFAULT_MAX_NEW_TOKENS = 256


class L20SmokeError(RuntimeError):
    """Raised for an actionable L20 smoke-test contract failure."""


def load_synthetic_fixture(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(FIXTURE_ROOT.resolve())
    except ValueError as exc:
        raise L20SmokeError(
            "only explicitly synthetic fixtures inside tests/fixtures may be used"
        ) from exc
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise L20SmokeError(f"cannot read synthetic fixture: {type(exc).__name__}") from exc
    if not isinstance(value, dict) or value.get("synthetic") is not True:
        raise L20SmokeError("fixture must be a JSON object with synthetic=true")
    patient_id = value.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.startswith("SYNTHETIC-"):
        raise L20SmokeError("synthetic fixture must use a SYNTHETIC- patient identifier")
    return value


def build_messages(patient: dict[str, Any], *, retry: bool) -> list[dict[str, str]]:
    schema = json.dumps(PREDICTION_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    fixture = json.dumps(patient, ensure_ascii=False, separators=(",", ":"))
    correction = (
        " Your previous response was not valid. Return one JSON object only, with no fence or prefix."
        if retry
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a software smoke-test DoctorAgent. The record is entirely fictional. "
                "Return a concise structured risk estimate for pipeline validation only. "
                "Do not provide chain-of-thought; reasoning_summary must be a short auditable conclusion."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fictional EHR fixture:\n{fixture}\n"
                f"Return JSON matching this JSON Schema exactly:\n{schema}."
                f"{correction}"
            ),
        },
    ]


def parse_prediction_text(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise L20SmokeError("model response is empty")
    start = content.find("{")
    if start < 0:
        raise L20SmokeError("model response contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(content, start)
    except json.JSONDecodeError as exc:
        raise L20SmokeError("model response JSON cannot be parsed") from exc
    if not isinstance(value, dict):
        raise L20SmokeError("model response JSON must be an object")
    try:
        return validate_prediction(value)
    except LLMResponseError as exc:
        raise L20SmokeError(f"model response violates required fields: {exc}") from exc


def parse_with_retry(
    generate_once: Callable[[int], tuple[str, dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    last_error: L20SmokeError | None = None
    for attempt in range(2):
        content, generation = generate_once(attempt)
        try:
            return parse_prediction_text(content), generation, attempt + 1
        except L20SmokeError as exc:
            last_error = exc
    raise L20SmokeError("model output was invalid after 2 attempts") from last_error


def enforce_context_limit(
    *, input_tokens: int, max_new_tokens: int, context_length: int
) -> None:
    if context_length <= 0 or context_length > FIRST_VERSION_CONTEXT_LIMIT:
        raise L20SmokeError(
            f"context length exceeds the first-version limit of {FIRST_VERSION_CONTEXT_LIMIT}"
        )
    if input_tokens <= 0 or max_new_tokens <= 0:
        raise L20SmokeError("input and output token counts must be positive")
    if max_new_tokens > DEFAULT_MAX_NEW_TOKENS:
        raise L20SmokeError(
            f"max_new_tokens exceeds the first-version limit of {DEFAULT_MAX_NEW_TOKENS}"
        )
    if input_tokens + max_new_tokens > context_length:
        raise L20SmokeError(
            f"input_tokens + max_new_tokens exceeds the configured {context_length} context"
        )


def _device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError as exc:
            raise L20SmokeError(f"invalid CUDA device: {device}") from exc
    raise L20SmokeError("the first L20 smoke test requires a CUDA device")


def _query_gpu_memory(index: int) -> dict[str, int]:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=15
        )
        fields = [int(value.strip()) for value in completed.stdout.strip().split(",")]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise L20SmokeError("nvidia-smi memory query failed") from exc
    if len(fields) != 2:
        raise L20SmokeError("nvidia-smi returned an unexpected memory result")
    return {"used_mib": fields[0], "total_mib": fields[1]}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mib(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise L20SmokeError(f"local model directory does not exist: {model_path}")
    for name in ("config.json", "tokenizer_config.json"):
        if not (model_path / name).is_file():
            raise L20SmokeError(f"local model directory is missing {name}")
    if not list(model_path.glob("*.safetensors")):
        raise L20SmokeError("local model directory has no safetensors weights")
    patient = load_synthetic_fixture(args.fixture)
    enforce_context_limit(
        input_tokens=1,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise L20SmokeError(f"required Python package is unavailable: {exc.name}") from exc

    device_index = _device_index(args.device)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= device_index:
        raise L20SmokeError(f"CUDA device is unavailable: {args.device}")
    if not torch.cuda.is_bf16_supported():
        raise L20SmokeError("the selected CUDA device does not support BF16")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    before_load = _query_gpu_memory(device_index)
    torch.cuda.empty_cache()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    after_load = _query_gpu_memory(device_index)
    torch.cuda.reset_peak_memory_stats(device)
    generation_records: list[dict[str, Any]] = []

    def generate_once(attempt: int) -> tuple[str, dict[str, Any]]:
        rendered = tokenizer.apply_chat_template(
            build_messages(patient, retry=attempt > 0),
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(
            [rendered], return_tensors="pt", add_special_tokens=False
        ).to(device)
        input_tokens = int(model_inputs["input_ids"].shape[-1])
        enforce_context_limit(
            input_tokens=input_tokens,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - started
        output_ids = generated[0, input_tokens:]
        output_tokens = int(output_ids.shape[-1])
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        record = {
            "attempt": attempt + 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_seconds": round(inference_seconds, 4),
        }
        generation_records.append(record)
        return content, record

    prediction, successful_generation, attempts = parse_with_retry(generate_once)
    after_inference = _query_gpu_memory(device_index)
    metrics = {
        "model_load_seconds": round(load_seconds, 4),
        "inference_seconds": round(
            sum(record["inference_seconds"] for record in generation_records), 4
        ),
        "input_tokens": successful_generation["input_tokens"],
        "output_tokens": successful_generation["output_tokens"],
        "attempts": attempts,
        "gpu_memory_before_load_mib": before_load,
        "gpu_memory_after_load_mib": after_load,
        "gpu_memory_after_inference_mib": after_inference,
        "inference_peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
        "inference_peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
    }
    result = {
        "status": "success",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fixture": args.fixture.name,
        "synthetic": True,
        "model_directory": model_path.name,
        "device": args.device,
        "dtype": "bfloat16",
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "prediction": prediction,
        "metrics": metrics,
    }
    _write_json_atomic(args.output_dir.resolve() / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=os.environ.get("MODEL_PATH")
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=FIRST_VERSION_CONTEXT_LIMIT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model_path is None:
        print("L20 Transformers smoke failed: set MODEL_PATH or --model-path", file=sys.stderr)
        return 2
    try:
        result = run_smoke(args)
    except Exception:
        print("L20 Transformers smoke failed with the following exception:", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

