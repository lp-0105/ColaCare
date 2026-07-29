#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the independent L20 repository}"
: "${DATA_ROOT:?Set DATA_ROOT to validated formal-v2/fold_1}"
: "${LOG_ROOT:?Set LOG_ROOT to a new stage log directory}"
: "${RESULTS_ROOT:?Set RESULTS_ROOT to a new results directory}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the uploaded bundle commit}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-42}"

mkdir -p "$LOG_ROOT" "$RESULTS_ROOT"
PROGRESS="$LOG_ROOT/progress.md"
COMMANDS="$LOG_ROOT/commands.log"

record() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$PROGRESS"
}

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" >>"$COMMANDS"
  printf '\n' >>"$COMMANDS"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

cd "$REPO_ROOT"
ACTUAL_BRANCH="$(git branch --show-current)"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_BRANCH" != "fix/adacare-batch-semantics" ]]; then
  echo "Unexpected branch: $ACTUAL_BRANCH" >&2
  exit 2
fi
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Unexpected commit: $ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --short)" ]]; then
  echo "L20 independent repository is not clean" >&2
  git status --short >&2
  exit 2
fi

{
  date --iso-8601=seconds
  uname -a
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c 'import numpy,pandas,psutil,sklearn,torch; print("numpy",numpy.__version__); print("pandas",pandas.__version__); print("psutil",psutil.__version__); print("sklearn",sklearn.__version__); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("cuda_available",torch.cuda.is_available()); print("gpu",torch.cuda.get_device_name(0)); print("bf16",torch.cuda.is_bf16_supported())'
  "$PYTHON_BIN" -c 'import shap; print("shap",shap.__version__)'
  nvidia-smi
  df -h "$RESULTS_ROOT"
  git branch --show-current
  git rev-parse HEAD
  git status --short
} >"$LOG_ROOT/environment.txt" 2>&1

record "START repository=$REPO_ROOT data=formal_v2 commit=$EXPECTED_COMMIT"
run_logged unit_tests "$PYTHON_BIN" -m pytest -q tests
run_logged compileall "$PYTHON_BIN" -m compileall -q pyehr utils scripts tests
run_logged data_validation "$PYTHON_BIN" scripts/validate_mimiciv_processed.py \
  --parquet "$(dirname "$DATA_ROOT")/mimic4_formatted_ehr.parquet" \
  --data-dir "$DATA_ROOT" \
  --output "$LOG_ROOT/formal_v2_validation_report.json"
record "PASS local-equivalent tests and formal-v2 validation"

GATE_DIR="$RESULTS_ROOT/adacare_outcome_seed42_batchfix_v2_gate1"
run_logged adacare_gate_train "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py train \
  --model AdaCare --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
  --output-dir "$GATE_DIR" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --learning-rate "$LEARNING_RATE" \
  --max-epochs 1 --patience 10 --bootstrap-iterations 0 --seed "$SEED"
run_logged adacare_gate_invariance "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py invariance \
  --model AdaCare --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
  --checkpoint "$GATE_DIR/best_checkpoint.pt" \
  --output "$GATE_DIR/invariance_tests.json" \
  --hidden-dim "$HIDDEN_DIM" --seed "$SEED"
record "PASS AdaCare formal-v2 one-epoch semantic gate"

RETAIN_DIR="$RESULTS_ROOT/retain_outcome_seed42_formal_v2"
CONCARE_DIR="$RESULTS_ROOT/concare_outcome_seed42_formal_v2"
ADACARE_DIR="$RESULTS_ROOT/adacare_outcome_seed42_batchfix_v2"

run_logged retain_formal "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py train \
  --model RETAIN --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
  --output-dir "$RETAIN_DIR" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --learning-rate "$LEARNING_RATE" \
  --max-epochs 50 --patience 10 --bootstrap-iterations 100 --seed "$SEED"
record "PASS RETAIN formal-v2 training"

run_logged concare_formal "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py train \
  --model ConCare --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
  --output-dir "$CONCARE_DIR" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --learning-rate "$LEARNING_RATE" \
  --max-epochs 50 --patience 10 --bootstrap-iterations 100 --seed "$SEED"
record "PASS ConCare formal-v2 training"

run_logged adacare_formal "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py train \
  --model AdaCare --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
  --output-dir "$ADACARE_DIR" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --learning-rate "$LEARNING_RATE" \
  --max-epochs 50 --patience 10 --bootstrap-iterations 100 --seed "$SEED"
run_logged adacare_formal_invariance "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py invariance \
  --model AdaCare --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
  --checkpoint "$ADACARE_DIR/best_checkpoint.pt" \
  --output "$ADACARE_DIR/invariance_tests.json" \
  --hidden-dim "$HIDDEN_DIM" --seed "$SEED"
record "PASS AdaCare formal-v2 batchfix training and invariance"

SHAP_ROOT="$RESULTS_ROOT/shap_smoke_32"
mkdir -p "$SHAP_ROOT"
run_logged sample_manifest "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py sample-manifest \
  --data-dir "$DATA_ROOT" --output "$SHAP_ROOT/sample_manifest.json" \
  --sample-count 32 --seed "$SEED"

for SPEC in \
  "RETAIN:$RETAIN_DIR" \
  "ConCare:$CONCARE_DIR" \
  "AdaCare:$ADACARE_DIR"
do
  MODEL="${SPEC%%:*}"
  MODEL_DIR="${SPEC#*:}"
  LOWER="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
  run_logged "${LOWER}_expert_smoke" "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py expert-smoke \
    --model "$MODEL" --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
    --checkpoint "$MODEL_DIR/best_checkpoint.pt" \
    --test-predictions "$MODEL_DIR/test_predictions.npz" \
    --sample-manifest "$SHAP_ROOT/sample_manifest.json" \
    --output-dir "$SHAP_ROOT/$LOWER/expert" \
    --hidden-dim "$HIDDEN_DIM" --batch-size 32 --seed "$SEED"
done
record "PASS three-model anonymous expert-output smoke"

for SPEC in \
  "RETAIN:$RETAIN_DIR" \
  "ConCare:$CONCARE_DIR" \
  "AdaCare:$ADACARE_DIR"
do
  MODEL="${SPEC%%:*}"
  MODEL_DIR="${SPEC#*:}"
  LOWER="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
  run_logged "${LOWER}_shap_smoke" "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py shap-smoke \
    --model "$MODEL" --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
    --checkpoint "$MODEL_DIR/best_checkpoint.pt" \
    --sample-manifest "$SHAP_ROOT/sample_manifest.json" \
    --output-dir "$SHAP_ROOT/$LOWER/shap" \
    --hidden-dim "$HIDDEN_DIM" --background-size 64 \
    --kmeans-clusters 32 --predict-batch-size 512 --seed "$SEED"
done
record "PASS three-model repository-contract KernelSHAP smoke"

EXPORT_ROOT="$RESULTS_ROOT/expert_outputs_v2"
for SPEC in \
  "RETAIN:$RETAIN_DIR" \
  "ConCare:$CONCARE_DIR" \
  "AdaCare:$ADACARE_DIR"
do
  MODEL="${SPEC%%:*}"
  MODEL_DIR="${SPEC#*:}"
  LOWER="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
  run_logged "${LOWER}_expert_export" "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py export \
    --model "$MODEL" --repo-root "$REPO_ROOT" --data-dir "$DATA_ROOT" \
    --checkpoint "$MODEL_DIR/best_checkpoint.pt" \
    --output-root "$EXPORT_ROOT" --hidden-dim "$HIDDEN_DIM" \
    --batch-size "$BATCH_SIZE" --chunk-size 1024 --seed "$SEED"
done
run_logged expert_alignment "$PYTHON_BIN" scripts/run_mimiciv_expert_stage.py align \
  --output-root "$EXPORT_ROOT"
record "PASS full ordinary expert outputs and cross-model alignment"
record "COMPLETE full SHAP intentionally not started"
