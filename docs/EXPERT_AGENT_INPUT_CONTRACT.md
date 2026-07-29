# Expert-to-DoctorAgent input contract

## Scope and status

`expert-agent-input-v1` is a deterministic adapter contract for the MIMIC-IV
`formal_v2` data and the aligned `expert_outputs_v2` exports. It prepares
privacy-reduced JSON and text that a DoctorAgent can consume in a later stage.
This repository stage does **not** call an LLM, DoctorAgent, MetaAgent, RAG, or
Fusion model.

The adapter is not a replacement for the expert models. It joins four already
aligned sources:

| Source | Meaning | Agent-facing use |
| --- | --- | --- |
| `raw_x` | Unnormalised 48 × 61 values; missing cells remain non-finite | The only source for displayed clinical values |
| `missing_mask` | 48 × 61, where `1=missing` and `0=observed` | Prevents an imputed model value from being described as observed |
| Expert logits/probabilities | Outcome-model estimates from RETAIN, ConCare, and fixed AdaCare | Model-specific risk estimates |
| Native importance | RETAIN/AdaCare `[48,61]`; ConCare `[60]` | Within-model Top-K evidence |

The following sources are deliberately excluded:

- Normalised/imputed `x` is used by the expert models but is never rendered as
  a clinical measurement.
- The 128-dimensional patient embeddings remain Fusion inputs and never enter
  the Agent JSON or prompt.
- `y`, `Outcome`, exported `labels`, test metrics, and future information never
  enter the Agent JSON or prompt.
- Patient, admission, stay, record identifiers, database positions, record
  timestamps, notes, and paths are not emitted.

## Evidence from the repository

- `utils/mimiciv_preprocess_utils.py::_encode_split` creates the exact model
  column order `Sex`, `Age`, 47 historical categorical one-hot columns, then 12
  numerical columns. It writes `x`, `raw_x`, `y`, `pid`, `missing_mask`, and
  `record_time` separately.
- In that function, categorical and numerical mask entries are `1` when the
  source is missing. Numerical model inputs are forward-filled and then filled
  with training-only medians, while `raw_x` retains missing values.
- `scripts/run_mimiciv_expert_stage.py::predict_in_batches` exports logits,
  probabilities, 128-dimensional embeddings, and each model's native feature
  importance. The full export archive also contains labels for offline
  evaluation. The adapter reads only an explicit whitelist and never copies
  labels or embeddings.
- The legacy `utils/healthcare_context_utils.py::ContextBuilder` builds the
  human-readable `healthcare_context` consumed by
  `utils/framework.py::DoctorAgent.analysis`. It does not distinguish imputed
  values robustly and expects notes that formal_v2 does not generate. The new
  deterministic text is the future safe replacement for that context; it is
  not wired into the Agent in this stage.

## JSON structure

Each file uses an anonymous ordinal such as `sample_000000`:

```json
{
  "schema_version": "expert-agent-input-v1",
  "anonymous_sample_id": "sample_000000",
  "split": "test",
  "observation_window_hours": 48,
  "synthetic_demo": false,
  "patient_summary": {},
  "expert_predictions": {},
  "expert_evidence": {},
  "agreement_summary": {},
  "provenance": {}
}
```

The split name is retained for experiment bookkeeping. The original
split-relative position is never written to per-sample JSON, text, or the
output manifest.

## Patient summary rules

The builder is deterministic and uses no model or free-text generation.

- Exact age is reduced to a configured band. Sex uses the source mapping
  `F=0`, `M=1`.
- Continuous summaries use observed cells only: first, latest, minimum,
  maximum, observed count, missing fraction, and
  `latest_observed - first_observed`.
- The trend direction is `increasing`, `decreasing`, or `stable` under the
  configured tolerance.
- Categorical states are decoded from the fixed one-hot groups. First state,
  latest state, state-change count, observed count, and missing fraction are
  emitted.
- A missing cell has no displayed raw value. Evidence may say that the expert
  model operated on an imputed input, but it may not call that value an
  observation.
- Real record times are excluded. Evidence time is a relative bin
  `time_hour=0..47`.

## Expert predictions and agreement

Probabilities and logits are copied without recalibration. The display-only
risk bands are:

- `<0.25`: lower model-estimated risk
- `0.25–<0.50`: moderate model-estimated risk
- `0.50–<0.75`: elevated model-estimated risk
- `>=0.75`: high model-estimated risk

These are research-model display intervals, not clinical grades. The
agreement summary is computed from the three probabilities using mean,
minimum, maximum, range, population standard deviation, deterministic ranking,
and fixed range thresholds:

- range `<=0.10`: low disagreement
- range `>0.10` and `<=0.25`: moderate disagreement
- range `>0.25`: high disagreement

No LLM assigns these fields.

## Expert evidence

The default `Top-K` is 8 and can be reduced or increased by CLI.

For RETAIN and AdaCare, feature ranking uses the sum of absolute native
importance over all 48 hours. The selected hour is the largest absolute
native value for that feature. The native value and the absolute ranking score
are reported separately.

ConCare uses its confirmed 60-position mapping and is described in
`docs/CONCARE_IMPORTANCE_SEMANTICS.md`. Its exported attention is already
time-aggregated, so `time_hour` and `raw_value` remain `null`; the adapter does
not invent a representative hour.

Native importance values are only ranked within their originating model. They
are not treated as a common scale across RETAIN, ConCare, and AdaCare.

## Privacy and leakage gates

`utils/privacy_validator.py` recursively rejects database identifier keys and
identifier-like text, target/label keys, patient-vector keys, and calendar
timestamps. Output generation is whitelist-based. A generated prompt must be
byte-identical when rebuilt from its JSON.

Every prompt includes these limitations:

- It covers only the first 48 hours of structured ICU data.
- Unobserved values are not described as real measurements.
- Probabilities are research-model outputs, not clinical diagnoses.
- Labels were not provided to the Agent.
- Patient identity information is absent.

## CLI

Build selected anonymous rows:

```bash
python scripts/build_expert_agent_inputs.py \
  --formal-fold-dir /path/to/formal_v2/fold_1 \
  --expert-output-dir /path/to/expert_outputs_v2 \
  --feature-mapping /path/to/formal_v2/feature_mapping.json \
  --split test \
  --sample-indices 0,1,2 \
  --output-dir /path/to/ignored/agent_inputs \
  --top-k 8 \
  --seed 42
```

Validate existing output without rebuilding:

```bash
python scripts/build_expert_agent_inputs.py \
  --output-dir /path/to/ignored/agent_inputs \
  --validate-only
```

The output directory contains `manifest.json`, one JSON and one deterministic
text file per anonymous sample, and `validation_report.json`. Generated
`artifacts/` and Agent input directories remain ignored by Git.
