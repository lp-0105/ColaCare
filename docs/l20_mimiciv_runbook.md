# L20 MIMIC-IV offline runbook

## Scope and current status

The local Windows stage has validated the following limited chain on a
600-episode, pseudonymized MIMIC-IV 2.2 smoke cohort:

```text
raw CSV.GZ -> formatted parquet -> subject-safe pickle splits
           -> RETAIN forward/backward/checkpoint/evaluation
```

The local RETAIN run used CUDA, batch size 16, hidden dimension 64, and three
epochs. Training loss changed from 0.6128 to 0.3728 and the validation/test
probabilities were finite and within `[0,1]`. These values are plumbing checks,
not paper metrics.

No Agent, RAG, Fusion, SHAP, formal embedding, or full-cohort training has been
run on these MIMIC-IV data.

The local formal preparation is now also complete: 23,653 first-ICU-per-subject
episodes, fixed `lab_dim=59`, subject-safe 90/5/5 pickles, and a one-epoch
RETAIN CUDA plumbing run. This still is not a paper-metric reproduction.

## Security boundary

Processed MIMIC data remain controlled patient-derived data even when direct
identifiers have been replaced. They must not be committed to GitHub, attached
to public issues, or placed in a model repository.

Use the school's approved transfer path. Keep all data under a protected L20
working directory. Do not include `.env`, credentials, model weights, logs with
patient-level content, or raw identifiers in a code archive.

## What to upload

### Code

Upload a code archive made from the eventual reviewed commit containing at
least:

```text
ehr_datasets/mimic-iv/preprocess.py
utils/mimiciv_preprocess_utils.py
scripts/prepare_mimiciv_smoke.py
scripts/train_mimiciv_ehr_smoke.py
tests/test_mimiciv_pipeline.py
pyehr/
requirements.txt
docs/MIMICIV_DATA_CONTRACT.md
docs/l20_mimiciv_runbook.md
```

The raw extraction script is useful for provenance, but is not required to
train from already generated pickle files.

### Processed data

For the first L20 validation upload:

```text
ehr_datasets/mimic-iv/processed/mimic4_formatted_ehr.parquet
ehr_datasets/mimic-iv/processed/formatted_ehr_summary.json
ehr_datasets/mimic-iv/processed/fold_1/
```

The current smoke processed directory contains 32 files and is approximately
16.35 MB. The `fold_1` directory is approximately 15.95 MB.

An upload-ready controlled-data archive was generated locally at:

```text
ehr_datasets/mimic-iv/processed/l20_smoke_package/l20-mimiciv-smoke-600.tar.gz
ehr_datasets/mimic-iv/processed/l20_smoke_package/l20-mimiciv-smoke-600.tar.gz.sha256
```

The archive is about 1.03 MiB compressed and contains 30 files / 16,231,534
input bytes. It remains MIMIC-derived controlled data and is Git-ignored.

The local smoke checkpoint may be uploaded only as an optional compatibility
artifact:

```text
ehr_datasets/mimic-iv/processed/smoke_results/retain_outcome_smoke.pt
```

It must not be treated as a formally trained checkpoint.

### Is raw MIMIC-IV required on L20?

No, not for training, evaluation, SHAP, or embedding when the parquet/pickle
cohort and feature contract are frozen. Uploading only processed data is the
preferred first step and reduces disk use and exposure.

Raw MIMIC-IV is required only when L20 must regenerate the cohort, change item
mappings/windows, audit raw joins, or perform a full raw extraction. Any such
upload remains subject to the MIMIC data use agreement and platform policy.

## Offline Python assets

The L20 already has Python 3.10, CUDA-enabled PyTorch, and Transformers. Do not
replace the shared PyTorch environment. Prefer cloning it to a personal path or
using a personal `--target` vendor directory.

For the direct RETAIN training script, prepare Linux x86_64/Python 3.10 wheels
for versions compatible with the L20 environment:

```text
numpy
pandas
psutil
python-dateutil
pytz
tzdata
```

For the repository Lightning training/evaluation path, additionally prepare:

```text
lightning==2.3.3
torchmetrics==1.4.0.post0
lightning-utilities
scikit-learn==1.5.1
scipy
joblib
threadpoolctl
matplotlib
ipdb==0.13.13
tqdm==4.66.4
PyYAML
fsspec
packaging
typing-extensions
```

For the later SHAP phase, also prepare `shap==0.46.0` and its resolved offline
dependencies such as numba and llvmlite. Build a wheel manifest and SHA256 file
on the connected machine and verify both after upload.

`pyarrow` is needed on L20 only if parquet must be read or preprocessing must
be rerun. Training directly from pickle files does not require it.

Example offline installation into a private vendor directory:

```bash
export ASSET_ROOT=/path/to/approved/offline/assets
export COLACARE_ROOT=/path/to/ColaCare
export VENDOR_ROOT=/path/to/private/vendor/mimiciv

mkdir -p "$VENDOR_ROOT"
python -m pip install \
  --no-index \
  --find-links "$ASSET_ROOT/wheels" \
  --target "$VENDOR_ROOT" \
  -r "$ASSET_ROOT/mimiciv-runtime-requirements.txt"

export PYTHONPATH="$VENDOR_ROOT:$COLACARE_ROOT:${PYTHONPATH:-}"
```

Do not run this against `/opt/miniconda3/envs/pytorch` without explicit
authorization.

## Expected L20 layout

```text
$WORK_ROOT/
├── ColaCare/
│   ├── pyehr/
│   ├── scripts/
│   ├── utils/
│   └── ehr_datasets/mimic-iv/processed/fold_1/
├── vendor/mimiciv/
├── wheels/
├── checkpoints/
└── logs/
```

Keep checkpoints and logs outside Git-tracked source paths.

## First L20 validation

Verify and unpack the controlled smoke package before running anything:

```bash
export ASSET_ROOT=/path/to/approved/upload
export WORK_ROOT=/path/to/protected/work

cd "$ASSET_ROOT"
sha256sum -c l20-mimiciv-smoke-600.tar.gz.sha256
tar -tzf l20-mimiciv-smoke-600.tar.gz
mkdir -p "$WORK_ROOT"
tar -xzf l20-mimiciv-smoke-600.tar.gz -C "$WORK_ROOT"
```

If the existing L20 environment lacks pandas, numpy, or psutil, install only
pre-downloaded Linux wheels into a personal target, never into the shared
Conda environment:

```bash
export VENDOR_ROOT=/path/to/private/vendor/mimiciv
mkdir -p "$VENDOR_ROOT"
python -m pip install --no-index --no-deps --target "$VENDOR_ROOT" \
  /path/to/wheels/*.whl
```

Run the one-command environment/data/train/reload check:

```bash
export COLACARE_ROOT=/path/to/ColaCare
export MIMIC_SMOKE_ROOT="$WORK_ROOT/mimiciv-smoke-600"
export OUTPUT_ROOT=/path/to/protected/results/mimiciv-smoke
export VENDOR_ROOT=/path/to/private/vendor/mimiciv
bash "$COLACARE_ROOT/scripts/run_l20_mimiciv_smoke.sh"
```

The script does not install packages or modify the shared environment. It
records environment versions, validates the three pickle splits, trains RETAIN
for one epoch, reloads the checkpoint, and records `nvidia-smi`.

The individual validation commands remain:

```bash
export COLACARE_ROOT=/path/to/ColaCare
export PYTHONPATH=/path/to/private/vendor/mimiciv:$COLACARE_ROOT:${PYTHONPATH:-}
cd "$COLACARE_ROOT"

python -m pytest -q tests/test_mimiciv_pipeline.py
python -m compileall -q utils scripts tests ehr_datasets/mimic-iv pyehr/models

python scripts/train_mimiciv_ehr_smoke.py \
  --data-dir ehr_datasets/mimic-iv/processed/fold_1 \
  --output-dir /path/to/checkpoints/mimiciv-smoke \
  --epochs 1 \
  --batch-size 16 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --seed 42
```

Acceptance criteria:

- CUDA device is the NVIDIA L20;
- checkpoint is written outside Git;
- one forward/backward pass and evaluation finish;
- probabilities are finite and in `[0,1]`;
- no subject overlap exists across splits;
- no network access is attempted.

## Scaling from smoke to a full RETAIN run

The generated formal `fold_1` is 873,774,006 bytes. Together with the formal
parquet and JSON provenance files, allow roughly 0.9 GB for upload and at least
3 GB protected working space; the 721 MB resumable SQLite raw-scan index is not
needed on L20. After the formal processed data have been uploaded:

```bash
python scripts/train_mimiciv_ehr_smoke.py \
  --data-dir /path/to/protected/mimiciv-formal/fold_1 \
  --output-dir /path/to/checkpoints/mimiciv-retain-full \
  --epochs 50 \
  --batch-size 128 \
  --hidden-dim 128 \
  --learning-rate 0.001 \
  --seed 42
```

Despite the script name, these arguments exercise a longer RETAIN training
run. This direct trainer does not yet replace the repository Lightning path for
formal embeddings, checkpoint selection, bootstrap metrics, or SHAP.

Before using `pyehr/train_test.py` formally, review its hard-coded
`devices=[1]` setting for a single-GPU L20 and constrain `pyehr/configs/hparams.py`
to the intended dataset/task/model. Do not launch the unmodified loop as a
formal experiment.

## Formal experiment stages still required

1. Reproduce the intended full raw-to-parquet benchmark cohort, using a full
   chartevents scan and a fixed categorical vocabulary.
2. Validate whether the expected full dynamic dimension is exactly 59.
3. Run AdaCare, ConCare/MCGRU discrepancy review, and RETAIN with fixed seeds.
4. Generate validation/test outputs and embeddings.
5. Run SHAP/feature-importance generation and verify output alignment.
6. Only then connect EHR outputs to DoctorAgent, MetaAgent, formal MedCPT/MSD
   RAG, GatorTron report embeddings, and Fusion.
7. Report AUROC/AUPRC/min(+P,Se) only after the full protocol is fixed.

## Resource estimates

- Smoke processed data: about 16.35 MB on disk and well below 1 GB RAM.
- Formal processed upload set: approximately 0.9 GB; preserve at least 3 GB
  for staging, checkpoints, metrics, and temporary copies.
- RETAIN smoke: local CUDA allocator peak was about 27.6 MB allocated and
  46.0 MB reserved; CUDA context overhead is not included.
- Formal batch-128 EHR training is expected to fit comfortably on an L20, but
  actual peak memory must be measured for each model and sequence distribution.
- The local formal RETAIN one-epoch run (batch 32, hidden 128) used about
  34.0 MiB CUDA allocated, 48.0 MiB CUDA reserved, and 1.74 GiB peak process
  RSS; the CUDA context/driver allocation shown by `nvidia-smi` may be higher.
- SHAP can be substantially slower and more memory-intensive than training;
  start with a small background and sample count before expansion.
