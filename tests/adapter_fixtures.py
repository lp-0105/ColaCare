from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "expert_agent_input_v1.json"
CONCARE_MAPPING_PATH = REPO_ROOT / "configs" / "concare_importance_mapping_v1.json"


def load_contracts() -> tuple[dict, dict]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    concare_mapping = json.loads(CONCARE_MAPPING_PATH.read_text(encoding="utf-8"))
    return config, concare_mapping


def synthetic_case(*, disagreement: bool = False, empty: bool = False) -> dict:
    config, concare_mapping = load_contracts()
    feature_names = config["feature_contract"]["feature_names"]
    raw_x = np.full((48, 61), np.nan, dtype=np.float32)
    missing_mask = np.ones((48, 61), dtype=np.float32)

    if not empty:
        raw_x[:, 0] = 1.0
        raw_x[:, 1] = 67.0
        missing_mask[:, :2] = 0.0

        capillary_indices = [2, 3]
        raw_x[0, capillary_indices] = [1.0, 0.0]
        raw_x[47, capillary_indices] = [0.0, 1.0]
        missing_mask[[0, 47], :][:, capillary_indices] = 0.0
        missing_mask[0, capillary_indices] = 0.0
        missing_mask[47, capillary_indices] = 0.0

        eye_indices = list(range(4, 12))
        raw_x[0, eye_indices] = 0.0
        raw_x[0, 4] = 1.0
        raw_x[47, eye_indices] = 0.0
        raw_x[47, 10] = 1.0
        missing_mask[0, eye_indices] = 0.0
        missing_mask[47, eye_indices] = 0.0

        heart_rate_index = feature_names.index("Heart Rate")
        for hour, value in ((0, 80.0), (24, 95.0), (47, 110.0)):
            raw_x[hour, heart_rate_index] = value
            missing_mask[hour, heart_rate_index] = 0.0

        glucose_index = feature_names.index("Glucose")
        raw_x[12, glucose_index] = 145.0
        missing_mask[12, glucose_index] = 0.0

    retain = np.zeros((48, 61), dtype=np.float32)
    adacare = np.zeros((48, 61), dtype=np.float32)
    concare = np.zeros(60, dtype=np.float32)
    heart_rate_index = feature_names.index("Heart Rate")
    glucose_index = feature_names.index("Glucose")
    retain[47, heart_rate_index] = 0.9
    retain[12, glucose_index] = -0.7
    retain[10, feature_names.index("pH")] = 0.6
    adacare[24, heart_rate_index] = 0.8
    adacare[5, glucose_index] = 0.5
    adacare[3, feature_names.index("Temperature")] = -0.4
    concare[heart_rate_index - 2] = 0.45
    concare[glucose_index - 2] = 0.30
    concare[59] = 0.20
    concare[0] = 0.05

    probabilities = (
        {"RETAIN": 0.10, "ConCare": 0.52, "AdaCare": 0.91}
        if disagreement
        else {"RETAIN": 0.31, "ConCare": 0.34, "AdaCare": 0.36}
    )
    expert_outputs = {}
    importance = {
        "RETAIN": retain,
        "ConCare": concare,
        "AdaCare": adacare,
    }
    for model_name in ("RETAIN", "ConCare", "AdaCare"):
        probability = probabilities[model_name]
        expert_outputs[model_name] = {
            "probability": probability,
            "logit": float(np.log(probability / (1.0 - probability))),
            "feature_importance": importance[model_name],
            "embeddings": np.arange(128, dtype=np.float32) + 1000.0,
            "labels": np.asarray([1], dtype=np.int64),
            "outcome": 1,
            "checkpoint_hash": hashlib.sha256(model_name.encode("utf-8")).hexdigest(),
        }

    return {
        "config": copy.deepcopy(config),
        "concare_mapping": copy.deepcopy(concare_mapping),
        "feature_names": feature_names,
        "raw_x": raw_x,
        "missing_mask": missing_mask,
        "expert_outputs": expert_outputs,
    }
