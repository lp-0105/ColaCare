from __future__ import annotations

from typing import Any

from utils.privacy_validator import validate_privacy


MODEL_ORDER = ("RETAIN", "ConCare", "AdaCare")


def _format_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def build_deterministic_prompt(payload: dict[str, Any]) -> str:
    """Render the adapter JSON into a stable DoctorAgent-readable text block."""

    validate_privacy(payload)
    summary = payload["patient_summary"]
    lines: list[str] = []
    if payload.get("synthetic_demo"):
        lines.append("SYNTHETIC DEMO — NOT REAL PATIENT DATA")
        lines.append("")
    lines.extend(
        [
            f"Anonymous sample: {payload['anonymous_sample_id']}",
            "",
            "1. Observation window and data completeness",
            (
                f"- Structured ICU data cover the first "
                f"{payload['observation_window_hours']} hours in 1-hour bins."
            ),
            (
                "- Dynamic missing fraction: "
                f"{summary['data_completeness']['dynamic_missing_fraction']:.6f}."
            ),
            f"- {summary['missingness_semantics']}.",
            "",
            "2. Demographics",
            f"- Age band: {_format_value(summary['age']['age_band'])} "
            f"({summary['age']['observed_status']}).",
            f"- Sex: {_format_value(summary['sex']['value'])} "
            f"({summary['sex']['observed_status']}).",
            "",
            "3. Vital signs and laboratory overview",
        ]
    )
    for name, item in summary["continuous_features"].items():
        unit = f" {item['unit']}" if item["unit"] else ""
        if item["observed_status"] == "missing":
            lines.append(
                f"- {name}: no observed measurement; missing fraction "
                f"{item['missing_fraction']:.6f}."
            )
        else:
            lines.append(
                f"- {name}: first {_format_value(item['first_observed'])}{unit}; "
                f"latest {_format_value(item['latest_observed'])}{unit}; "
                f"range {_format_value(item['min_observed'])}–"
                f"{_format_value(item['max_observed'])}{unit}; "
                f"trend {item['trend']['direction']} "
                f"(latest-minus-earliest={_format_value(item['trend']['delta'])}); "
                f"observed {item['observed_count']} times; missing fraction "
                f"{item['missing_fraction']:.6f}."
            )

    lines.extend(["", "4. Categorical and GCS state changes"])
    for name, item in summary["categorical_features"].items():
        if item["observed_status"] == "missing":
            lines.append(f"- {name}: no observed state.")
        else:
            lines.append(
                f"- {name}: first {item['first_observed_state']}; "
                f"latest {item['latest_observed_state']}; "
                f"observed state changes {item['observed_state_changes']}; "
                f"observed {item['observed_count']} times; missing fraction "
                f"{item['missing_fraction']:.6f}."
            )

    lines.extend(["", "5. Research-model risk predictions"])
    for model_name in MODEL_ORDER:
        item = payload["expert_predictions"][model_name]
        lines.append(
            f"- {model_name}: probability {item['probability']:.6f}; "
            f"{item['risk_band']}."
        )

    lines.extend(["", "6. Within-model expert evidence"])
    for model_name in MODEL_ORDER:
        model_evidence = payload["expert_evidence"][model_name]
        lines.append(
            f"- {model_name} semantics: {model_evidence['importance_semantics']}."
        )
        for rank, item in enumerate(model_evidence["top_evidence"], start=1):
            time_text = (
                "time aggregated"
                if item["time_hour"] is None
                else f"hour {item['time_hour']}"
            )
            raw_text = _format_value(item["raw_value"])
            unit = f" {item['unit']}" if item["unit"] else ""
            lines.append(
                f"  {rank}. {item['feature_name']} ({item['feature_group']}), "
                f"{time_text}, importance {_format_value(item['importance'])}, "
                f"status {item['observed_status']}, raw value {raw_text}{unit}."
            )

    agreement = payload["agreement_summary"]
    lines.extend(
        [
            "",
            "7. Model agreement and disagreement",
            f"- Mean probability: {agreement['mean_probability']:.6f}.",
            f"- Probability range: {agreement['probability_range']:.6f}.",
            f"- Standard deviation: {agreement['standard_deviation']:.6f}.",
            f"- Ranking: {' > '.join(agreement['model_ranking'])}.",
            f"- Deterministic disagreement level: {agreement['disagreement_level']}.",
            "",
            "8. Fixed limitations",
            "- Content is based only on structured records from the first 48 hours after ICU admission.",
            "- Unobserved values are not described as real measurements.",
            "- The three probabilities are research-model outputs, not clinical diagnoses.",
            "- Labels were not provided to the Agent.",
            "- No patient identity information is included.",
        ]
    )
    return "\n".join(lines) + "\n"
