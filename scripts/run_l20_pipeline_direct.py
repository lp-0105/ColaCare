"""Run staged synthetic ColaCare agents with one reused offline Transformers model."""

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
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_synthetic_smoke import load_patient
from utils.llm_client import LLMResponse, LLMResponseError
from utils.local_rag import load_synthetic_guidelines
from utils.prediction_schema import validate_prediction
from utils.smoke_agents import (
    run_discussion_stage,
    run_meta_stage,
    run_rag_stage,
    run_single_stage,
    run_two_doctor_stage,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_patient.json"
DEFAULT_GUIDELINES = ROOT / "tests" / "fixtures" / "synthetic_guidelines.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "l20_pipeline"
ACCEPTANCE_STAGES = ("single", "two-doctors", "meta", "discussion", "rag")


def validate_stage_order(stages: list[str]) -> list[str]:
    if not stages or tuple(stages) != ACCEPTANCE_STAGES[: len(stages)]:
        raise ValueError(
            "stages must be a non-empty prefix of the strict acceptance order: "
            + ", ".join(ACCEPTANCE_STAGES)
        )
    return list(stages)


def messages_with_schema(
    messages: list[dict[str, str]], response_schema: Mapping[str, Any] | None
) -> list[dict[str, str]]:
    prepared = [dict(message) for message in messages]
    if response_schema is None:
        return prepared
    contract = (
        "\nReturn exactly one JSON object matching this JSON Schema. "
        "Do not use markdown fences, prefixes, or extra keys:\n"
        + json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
    )
    for index in range(len(prepared) - 1, -1, -1):
        if prepared[index].get("role") == "user":
            prepared[index]["content"] = prepared[index].get("content", "") + contract
            break
    else:
        prepared.append({"role": "user", "content": contract.lstrip()})
    return prepared


def extract_json_object(content: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise LLMResponseError("direct Transformers response is not text")
    start = content.find("{")
    if start < 0:
        raise LLMResponseError("direct Transformers response contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(content, start)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("direct Transformers response JSON cannot be parsed") from exc
    if not isinstance(value, dict):
        raise LLMResponseError("direct Transformers JSON response must be an object")
    return value


def summarize_calls(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "llm_call_count": len(records),
        "input_tokens": sum(int(record["input_tokens"]) for record in records),
        "output_tokens": sum(int(record["output_tokens"]) for record in records),
        "inference_seconds": round(
            sum(float(record["inference_seconds"]) for record in records), 4
        ),
    }


def _device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device: {device}") from exc
    raise ValueError("direct Transformers backend requires a CUDA device")


def _query_gpu_memory(index: int) -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    values = [int(value.strip()) for value in completed.stdout.strip().split(",")]
    if len(values) != 2:
        raise RuntimeError("nvidia-smi returned an unexpected memory result")
    return {"used_mib": values[0], "total_mib": values[1]}


def _mib(value: int) -> float:
    return round(value / (1024 * 1024), 2)


class DirectTransformersClient:
    """LLMClient-compatible adapter backed by one local Transformers model instance."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        context_length: int = 4096,
        max_new_tokens: int = 256,
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.model_path = model_path.expanduser().resolve()
        self.device_name = device
        self.device_index = _device_index(device)
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.calls: list[dict[str, Any]] = []
        self.last_raw_response: str | None = None
        if not self.model_path.is_dir():
            raise ValueError(f"model directory does not exist: {self.model_path}")
        if context_length <= 0 or context_length > 4096:
            raise ValueError("context_length must be between 1 and 4096")
        if max_new_tokens <= 0 or max_new_tokens > 256:
            raise ValueError("max_new_tokens must be between 1 and 256")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() <= self.device_index:
            raise RuntimeError(f"CUDA device is unavailable: {device}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected CUDA device does not support BF16")
        self.torch = torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        self.gpu_memory_before_load_mib = _query_gpu_memory(self.device_index)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        torch.cuda.synchronize(self.device)
        self.model_load_seconds = round(time.perf_counter() - started, 4)
        self.gpu_memory_after_load_mib = _query_gpu_memory(self.device_index)

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        rendered = self.tokenizer.apply_chat_template(
            messages_with_schema(messages, response_schema),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [rendered], return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens + self.max_new_tokens > self.context_length:
            raise LLMResponseError(
                f"input plus output exceeds {self.context_length} token context"
            )
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.torch.cuda.synchronize(self.device)
        inference_seconds = round(time.perf_counter() - started, 4)
        output_ids = generated[0, input_tokens:]
        output_tokens = int(output_ids.shape[-1])
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        self.last_raw_response = raw
        self.calls.append(
            {
                "call": len(self.calls) + 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "inference_seconds": inference_seconds,
            }
        )
        value = extract_json_object(raw)
        return LLMResponse(
            content=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.rstrip() + "\n")


def _run_stage(
    stage: str,
    patient: Mapping[str, Any],
    client: DirectTransformersClient,
    guidelines: Path,
) -> dict[str, Any]:
    if stage == "single":
        return run_single_stage(patient, client)
    if stage == "two-doctors":
        return run_two_doctor_stage(patient, client)
    if stage == "meta":
        return run_meta_stage(patient, client)
    if stage == "discussion":
        return run_discussion_stage(patient, client)
    if stage == "rag":
        return run_rag_stage(patient, client, load_synthetic_guidelines(guidelines))
    raise ValueError(f"unsupported stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--guidelines", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--stages", nargs="+", default=list(ACCEPTANCE_STAGES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = validate_stage_order(args.stages)
    if args.model_path is None:
        print("MODEL_PATH or --model-path is required", file=sys.stderr)
        return 2
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    patient = load_patient(args.fixture)
    command = (
        f"{sys.executable} scripts/run_l20_pipeline_direct.py --stages "
        + " ".join(stages)
    )
    try:
        client = DirectTransformersClient(
            args.model_path,
            device=args.device,
            context_length=args.context_length,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception:
        failure = traceback.format_exc()
        _append_log(output_root / stages[0] / "run.log", "status=failed phase=model_load")
        _append_log(output_root / stages[0] / "run.log", failure)
        print(failure, file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "backend": "direct-transformers",
                "model_load_count": 1,
                "model_load_seconds": client.model_load_seconds,
                "gpu_memory_before_load_mib": client.gpu_memory_before_load_mib,
                "gpu_memory_after_load_mib": client.gpu_memory_after_load_mib,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for stage in stages:
        stage_dir = output_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_log = stage_dir / "run.log"
        started_at = datetime.now(timezone.utc).isoformat()
        _append_log(stage_log, f"{started_at} status=started stage={stage}")
        _append_log(stage_log, f"command={command}")
        first_call = len(client.calls)
        stage_started = time.perf_counter()
        client.torch.cuda.reset_peak_memory_stats(client.device)
        gpu_before_stage = _query_gpu_memory(client.device_index)
        try:
            result = _run_stage(stage, patient, client, args.guidelines)
            validate_prediction(result["final_prediction"])
            stage_wall_seconds = round(time.perf_counter() - stage_started, 4)
            records = client.calls[first_call:]
            call_summary = summarize_calls(records)
            metrics = {
                "run_command": command,
                "backend": "direct-transformers",
                "model_path": str(client.model_path),
                "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                "device": args.device,
                "dtype": "bfloat16",
                "context_length": args.context_length,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "offline": True,
                "model_reuse": "one shared model instance for all requested stages",
                "model_load_count": 1,
                "model_load_seconds": client.model_load_seconds,
                **call_summary,
                "stage_wall_seconds": stage_wall_seconds,
                "gpu_memory_before_model_load_mib": client.gpu_memory_before_load_mib,
                "gpu_memory_after_model_load_mib": client.gpu_memory_after_load_mib,
                "gpu_memory_before_stage_mib": gpu_before_stage,
                "gpu_memory_after_inference_mib": _query_gpu_memory(client.device_index),
                "stage_peak_allocated_mib": _mib(
                    client.torch.cuda.max_memory_allocated(client.device)
                ),
                "stage_peak_reserved_mib": _mib(
                    client.torch.cuda.max_memory_reserved(client.device)
                ),
                "calls": records,
            }
            result["l20_transformers"] = metrics
            result_path = stage_dir / "result.json"
            write_json_atomic(result_path, result)
            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            if persisted.get("stage") != stage:
                raise RuntimeError("persisted result failed stage verification")
            _append_log(
                stage_log,
                json.dumps(
                    {"status": "success", "stage": stage, "metrics": metrics},
                    ensure_ascii=False,
                ),
            )
            print(
                json.dumps(
                    {
                        "status": "success",
                        "stage": stage,
                        "result": str(result_path),
                        "log": str(stage_log),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception:
            failure = traceback.format_exc()
            partial = summarize_calls(client.calls[first_call:])
            _append_log(
                stage_log,
                json.dumps(
                    {"status": "failed", "stage": stage, "partial_metrics": partial},
                    ensure_ascii=False,
                ),
            )
            _append_log(stage_log, failure)
            print(f"PIPELINE_STAGE_FAILED={stage}", file=sys.stderr, flush=True)
            print(failure, file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
