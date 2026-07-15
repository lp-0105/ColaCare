"""Run staged ColaCare roles on one entirely fictional EHR record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.llm_client import LLMClient
from utils.local_rag import load_synthetic_guidelines
from utils.smoke_agents import (
    SmokePipelineError,
    run_discussion_stage,
    run_meta_stage,
    run_rag_stage,
    run_single_stage,
    run_two_doctor_stage,
    validate_synthetic_patient,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_patient.json"
DEFAULT_GUIDELINES = ROOT / "tests" / "fixtures" / "synthetic_guidelines.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "smoke"


def load_patient(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokePipelineError(f"cannot load synthetic fixture: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise SmokePipelineError("synthetic fixture root must be a JSON object")
    return validate_synthetic_patient(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["single", "two-doctors", "meta", "discussion", "rag"],
        default="single",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--guidelines", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        patient = load_patient(args.fixture)
        client = LLMClient.from_env()
        if args.stage == "single":
            result = run_single_stage(patient, client)
        elif args.stage == "two-doctors":
            result = run_two_doctor_stage(patient, client)
        elif args.stage == "meta":
            result = run_meta_stage(patient, client)
        elif args.stage == "discussion":
            result = run_discussion_stage(patient, client)
        else:
            result = run_rag_stage(
                patient, client, load_synthetic_guidelines(args.guidelines)
            )
        result["llm"] = {
            "base_url": client.settings.base_url,
            "model": client.settings.model_name,
            "context_length": client.settings.context_length,
            "max_tokens": client.settings.max_tokens,
            "temperature": client.settings.temperature,
            "reasoning_effort": client.settings.reasoning_effort,
        }
        stage_dir = args.output_root / args.stage
        write_json_atomic(stage_dir / "result.json", result)
        stage_dir.mkdir(parents=True, exist_ok=True)
        with (stage_dir / "run.log").open("a", encoding="utf-8") as log:
            timestamp = datetime.now(timezone.utc).isoformat()
            log.write(f"{timestamp} status=success stage={args.stage} patient=synthetic\n")
    except (SmokePipelineError, OSError, ValueError) as exc:
        print(f"Synthetic smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
