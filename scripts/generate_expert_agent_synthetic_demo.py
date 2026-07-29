from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_expert_agent_inputs import _write_outputs
from utils.expert_agent_adapter import MODEL_NAMES, build_expert_agent_input, load_json


DEFAULT_CONFIG = REPO_ROOT / "configs" / "expert_agent_input_v1.json"
DEFAULT_CONCARE_MAPPING = (
    REPO_ROOT / "configs" / "concare_importance_mapping_v1.json"
)


def _synthetic_source(
    case_number: int, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    names = config["feature_contract"]["feature_names"]
    raw = np.full((48, 61), np.nan, dtype=np.float32)
    mask = np.ones((48, 61), dtype=np.float32)
    if case_number != 2:
        raw[:, 0] = float(case_number % 2)
        raw[:, 1] = 42.0 + 25.0 * case_number
        mask[:, :2] = 0.0

        capillary = [names.index("Capillary refill rate->0.0"), names.index("Capillary refill rate->1.0")]
        raw[0, capillary] = [1.0, 0.0]
        raw[47, capillary] = [0.0, 1.0]
        mask[0, capillary] = 0.0
        mask[47, capillary] = 0.0

        heart_rate = names.index("Heart Rate")
        glucose = names.index("Glucose")
        temperature = names.index("Temperature")
        for hour, value in ((0, 72.0), (16, 89.0), (47, 101.0 + case_number * 8)):
            raw[hour, heart_rate] = value
            mask[hour, heart_rate] = 0.0
        raw[12, glucose] = 118.0 + case_number * 40
        mask[12, glucose] = 0.0
        raw[20, temperature] = 37.0 + case_number * 0.7
        mask[20, temperature] = 0.0

    probabilities_by_case = (
        {"RETAIN": 0.29, "ConCare": 0.32, "AdaCare": 0.34},
        {"RETAIN": 0.12, "ConCare": 0.55, "AdaCare": 0.88},
        {"RETAIN": 0.40, "ConCare": 0.41, "AdaCare": 0.39},
    )
    temporal = {}
    for model_name, scale in (("RETAIN", 1.0), ("AdaCare", 0.8)):
        values = np.zeros((48, 61), dtype=np.float32)
        values[47, names.index("Heart Rate")] = 0.9 * scale
        values[12, names.index("Glucose")] = -0.6 * scale
        values[20, names.index("Temperature")] = 0.4 * scale
        temporal[model_name] = values
    concare = np.zeros(60, dtype=np.float32)
    concare[names.index("Heart Rate") - 2] = 0.42
    concare[names.index("Glucose") - 2] = 0.31
    concare[59] = 0.19
    concare[0] = 0.08

    outputs: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        probability = probabilities_by_case[case_number][model_name]
        outputs[model_name] = {
            "probability": probability,
            "logit": float(np.log(probability / (1.0 - probability))),
            "feature_importance": (
                concare if model_name == "ConCare" else temporal[model_name]
            ),
            "checkpoint_hash": hashlib.sha256(
                f"SYNTHETIC-{model_name}".encode("utf-8")
            ).hexdigest(),
        }
    return raw, mask, outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate three entirely synthetic expert-to-Agent demonstrations."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--concare-mapping", type=Path, default=DEFAULT_CONCARE_MAPPING
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_json(args.config)
        concare_mapping = load_json(args.concare_mapping)
        payloads = []
        for ordinal in range(3):
            raw, mask, outputs = _synthetic_source(ordinal, config)
            payloads.append(
                build_expert_agent_input(
                    raw_x=raw,
                    missing_mask=mask,
                    expert_outputs=outputs,
                    anonymous_sample_id=f"sample_{ordinal:06d}",
                    split="test",
                    config=config,
                    concare_mapping=concare_mapping,
                    top_k=args.top_k,
                    synthetic=True,
                    seed=args.seed,
                )
            )
        manifest = _write_outputs(
            output_dir=args.output_dir,
            payloads=payloads,
            split="test",
            seed=args.seed,
            top_k=args.top_k,
            synthetic=True,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"DEMO FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
