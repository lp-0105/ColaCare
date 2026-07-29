from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_expert_agent_inputs import validate_input_directory
from utils.deterministic_prompt_builder import build_deterministic_prompt
from utils.expert_agent_adapter import (
    MODEL_NAMES,
    build_expert_agent_input,
    load_json,
    sha256_file,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "expert_agent_input_v1.json"
DEFAULT_CONCARE_MAPPING = (
    REPO_ROOT / "configs" / "concare_importance_mapping_v1.json"
)
REQUIRED_EXPERT_ARRAYS = {
    "logits",
    "probabilities",
    "embeddings",
    "feature_importance",
    "anonymous_sample_indices",
}


def _parse_sample_indices(value: str) -> list[int]:
    path = Path(value)
    if path.is_file():
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            parsed = parsed.get("sample_indices", parsed.get("split_positions"))
        values = parsed
    else:
        values = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    if not isinstance(values, list):
        raise ValueError("sample indices must be a JSON list or comma-separated values")
    indices = [int(item) for item in values]
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("sample indices must be unique non-negative integers")
    return indices


def _load_pickle_rows(path: Path, indices: list[int]) -> list[np.ndarray]:
    with path.open("rb") as handle:
        values = pickle.load(handle)
    if max(indices) >= len(values):
        raise IndexError(f"sample index exceeds {path.name} length")
    return [np.asarray(values[index]) for index in indices]


def _load_model_rows(
    expert_root: Path,
    split: str,
    model_name: str,
    indices: list[int],
) -> tuple[list[dict[str, Any]], str]:
    model_key = model_name.lower()
    manifest_path = expert_root / f"{model_key}_export_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("model") != model_name
        or manifest.get("data_version") != "formal_v2"
        or manifest.get("contains_database_ids") is not False
    ):
        raise ValueError(f"{model_name} export manifest failed its contract gate")
    checkpoint_hash = str(manifest.get("checkpoint_sha256", "")).lower()
    wanted = set(indices)
    rows: dict[int, dict[str, Any]] = {}
    for chunk_path in sorted((expert_root / split / model_key).glob("chunk_*.npz")):
        with np.load(chunk_path, allow_pickle=False) as chunk:
            missing = REQUIRED_EXPERT_ARRAYS - set(chunk.files)
            if missing:
                raise ValueError(f"{chunk_path.name} missing arrays: {sorted(missing)}")
            anonymous_indices = np.asarray(
                chunk["anonymous_sample_indices"], dtype=np.int64
            )
            for offset, split_index in enumerate(anonymous_indices.tolist()):
                if split_index not in wanted:
                    continue
                if split_index in rows:
                    raise ValueError(
                        f"duplicate anonymous split position for {model_name}"
                    )
                rows[split_index] = {
                    "logit": float(chunk["logits"][offset]),
                    "probability": float(chunk["probabilities"][offset]),
                    "feature_importance": np.asarray(
                        chunk["feature_importance"][offset]
                    ),
                    "checkpoint_hash": checkpoint_hash,
                }
                # labels and embeddings may be present in the source archive but
                # are deliberately not copied into the Agent-facing row.
    missing_rows = [index for index in indices if index not in rows]
    if missing_rows:
        raise ValueError(
            f"{model_name} missing {len(missing_rows)} requested anonymous rows"
        )
    return [rows[index] for index in indices], checkpoint_hash


def _validate_feature_mapping(
    feature_mapping_path: Path, config: dict[str, Any]
) -> list[str]:
    mapping = load_json(feature_mapping_path)
    dynamic = [
        *mapping.get("categorical_columns", []),
        *mapping.get("numerical_features", []),
    ]
    names = ["Sex", "Age", *dynamic]
    if names != config["feature_contract"]["feature_names"]:
        raise ValueError("feature mapping does not match the exact formal_v2 contract")
    return names


def _write_outputs(
    *,
    output_dir: Path,
    payloads: list[dict[str, Any]],
    split: str,
    seed: int,
    top_k: int,
    synthetic: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = []
    for payload in payloads:
        sample_id = payload["anonymous_sample_id"]
        sample_ids.append(sample_id)
        json_path = output_dir / f"{sample_id}.json"
        text_path = output_dir / f"{sample_id}.txt"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        text_path.write_text(
            build_deterministic_prompt(payload),
            encoding="utf-8",
        )
    manifest = {
        "status": "PASS",
        "schema_version": "expert-agent-input-v1",
        "data_contract": "formal_v2",
        "split": split,
        "samples": len(payloads),
        "anonymous_sample_ids": sample_ids,
        "top_k": top_k,
        "seed": seed,
        "synthetic_demo": synthetic,
        "contains_database_ids": False,
        "contains_targets": False,
        "contains_fusion_vectors": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_input_directory(output_dir)
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if validation["status"] != "PASS":
        raise RuntimeError(f"generated output failed validation: {validation['errors']}")
    manifest["validation_report_sha256"] = sha256_file(
        output_dir / "validation_report.json"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build privacy-safe, label-free DoctorAgent inputs from formal_v2 "
            "raw tensors and aligned expert outputs."
        )
    )
    parser.add_argument("--formal-fold-dir", type=Path)
    parser.add_argument("--expert-output-dir", type=Path)
    parser.add_argument("--feature-mapping", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--concare-mapping", type=Path, default=DEFAULT_CONCARE_MAPPING
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Mark outputs as synthetic; this does not synthesize input tensors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            report = validate_input_directory(args.output_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "PASS" else 1
        required = {
            "--formal-fold-dir": args.formal_fold_dir,
            "--expert-output-dir": args.expert_output_dir,
            "--feature-mapping": args.feature_mapping,
            "--sample-indices": args.sample_indices,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing required build arguments: {', '.join(missing)}")

        config = load_json(args.config)
        concare_mapping = load_json(args.concare_mapping)
        _validate_feature_mapping(args.feature_mapping, config)
        indices = _parse_sample_indices(args.sample_indices)
        raw_rows = _load_pickle_rows(
            args.formal_fold_dir / f"{args.split}_raw_x.pkl", indices
        )
        mask_rows = _load_pickle_rows(
            args.formal_fold_dir / f"{args.split}_missing_mask.pkl", indices
        )
        model_rows = {}
        for model_name in MODEL_NAMES:
            model_rows[model_name], _ = _load_model_rows(
                args.expert_output_dir,
                args.split,
                model_name,
                indices,
            )
        top_k = int(args.top_k or config["default_top_k"])
        payloads = []
        for ordinal, (raw_x, missing_mask) in enumerate(zip(raw_rows, mask_rows)):
            expert_outputs = {
                model_name: model_rows[model_name][ordinal]
                for model_name in MODEL_NAMES
            }
            payloads.append(
                build_expert_agent_input(
                    raw_x=raw_x,
                    missing_mask=missing_mask,
                    expert_outputs=expert_outputs,
                    anonymous_sample_id=f"sample_{ordinal:06d}",
                    split=args.split,
                    config=config,
                    concare_mapping=concare_mapping,
                    top_k=top_k,
                    synthetic=args.synthetic,
                    seed=args.seed,
                )
            )
        manifest = _write_outputs(
            output_dir=args.output_dir,
            payloads=payloads,
            split=args.split,
            seed=args.seed,
            top_k=top_k,
            synthetic=args.synthetic,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"BUILD FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
