from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.deterministic_prompt_builder import build_deterministic_prompt
from utils.privacy_validator import PrivacyViolation, validate_privacy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input_directory(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = manifest.get("anonymous_sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("manifest anonymous_sample_ids must be a non-empty list")
    expected = [f"sample_{index:06d}" for index in range(len(sample_ids))]
    if sample_ids != expected:
        raise ValueError("anonymous sample IDs must be contiguous and deterministic")

    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for sample_id in sample_ids:
        json_path = input_dir / f"{sample_id}.json"
        text_path = input_dir / f"{sample_id}.txt"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("anonymous_sample_id") != sample_id:
                raise ValueError(f"{sample_id} payload ID mismatch")
            validate_privacy(payload)
            expected_text = build_deterministic_prompt(payload)
            actual_text = text_path.read_text(encoding="utf-8")
            if actual_text != expected_text:
                raise ValueError(f"{sample_id} prompt is not deterministic")
            files.extend(
                [
                    {
                        "relative_path": json_path.name,
                        "bytes": json_path.stat().st_size,
                        "sha256": _sha256(json_path),
                    },
                    {
                        "relative_path": text_path.name,
                        "bytes": text_path.stat().st_size,
                        "sha256": _sha256(text_path),
                    },
                ]
            )
        except (OSError, ValueError, KeyError, PrivacyViolation, json.JSONDecodeError) as error:
            errors.append(f"{sample_id}: {type(error).__name__}: {error}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "schema_version": "expert-agent-input-v1",
        "samples": len(sample_ids),
        "anonymous_sample_ids": sample_ids,
        "files": files,
        "privacy_gate": "PASS" if not errors else "FAIL",
        "determinism_gate": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate privacy-safe expert-to-Agent JSON and prompt files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_input_directory(args.input_dir)
        report_path = args.input_dir / "validation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        print(f"VALIDATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
