# MIMIC-IV data contract for ColaCare

This document separates settings stated by the paper, settings enforced by the
repository, and provisional choices made only for the local smoke test. The
local smoke test is not a reproduction of the paper metrics.

## Paper-defined settings

- MIMIC-IV provides ICU demographics and laboratory/vital-sign time series.
- The outcome task is binary: `0` means alive and `1` means deceased.
- The readmission task predicts readmission within 30 days after discharge.
- The reported MIMIC-IV split is approximately 90%/5%/5% with 17,397/966/968
  samples. The reported outcome-positive rates are approximately 11.9%.
- EHR baselines include AdaCare, ConCare, and RETAIN. Training uses AdamW,
  batch size 128, 50 epochs, and early stopping on validation AUPRC.
- The paper says it follows established EHR benchmark preprocessing, but does
  not specify the ICU episode rule, observation window, or time-bin width.

## Repository-defined settings

- `pyehr/ehrdatasets/loader/unpad.py` reads the final time step and uses
  `y[..., 0]` for `outcome` and `y[..., 2]` for `readmission`.
- The original `ehr_datasets/mimic-iv/preprocess.py` expected an upstream
  `mimic4_formatted_ehr.parquet`; the repository did not contain the raw-table
  extractor that creates it.
- The formatted schema is:

  - identifiers/time: `RecordID`, `PatientID`, `RecordTime`;
  - targets: `Outcome`, `LOS`, `Readmission`;
  - static variables: `Sex`, `Age`;
  - categorical dynamics: capillary refill, GCS eye, motor, total, verbal;
  - numerical dynamics: diastolic pressure, FiO2, glucose, heart rate, height,
    mean pressure, oxygen saturation, respiratory rate, systolic pressure,
    temperature, weight, and pH.

- MIMIC-IV hparams declare `demo_dim=2` and `lab_dim=59` in
  `pyehr/configs/hparams.py`. `pyehr/pipelines/dl_pipeline.py` adds these to
  form a 61-dimensional model input.
- The exact origin of 59 is recoverable from repository history. In
  `8548787^:ehr_datasets/mimic-iv/preprocess.py`, the MIMIC feature order is
  hard-coded as 47 one-hot columns for five categorical variables followed by
  the same 12 numerical variables. `pyehr/importance.py` independently slices
  MIMIC SHAP values after `demo_dim + 47`, confirming the same boundary.
- The current tree contains no released processed feature pickle or checkpoint
  metadata, but the historical 47 names are complete. They are now frozen in
  `COLACARE_59_CATEGORICAL_LEVELS`; no dimensions were guessed.
- The checked-in MIMIC-IV hparams currently select AdaCare, MCGRU, and RETAIN,
  although the paper discusses ConCare rather than MCGRU in its three-model
  baseline description.
- Pure EHR model training does not require clinical notes. The later
  collaboration context path attempts to read `val_notes.pkl`/`test_notes.pkl`;
  those files are deliberately not fabricated by this pipeline.

## Provisional local smoke contract

These choices exist to validate the local data path and must be confirmed
against the intended benchmark before a formal run:

- one episode is the first ICU stay of a hospital admission;
- adults only (`Age >= 18`), ICU LOS at least 48 hours;
- at most one episode per subject for this 600-episode smoke cohort;
- first 48 ICU hours, one-hour bins;
- outcome is `admissions.hospital_expire_flag`;
- readmission is derived from the next admission beginning within 30 days of
  discharge, but is not evaluated in this outcome smoke test;
- raw identifiers are used only in memory for filtering, then replaced by
  sequential `PatientID`/`RecordID` pseudonyms before parquet is written;
- the GCS total is taken from a source item when present, otherwise derived
  from same-hour eye, motor, and verbal components;
- `chartevents.csv.gz` is read in chunks. The smoke command may stop when the
  observed subject ordering has passed all candidates. A formal full-cohort
  extraction must use `--full-chartevents-scan`.

## Corrected pickle contract

The repaired preprocessing path writes directly to
`ehr_datasets/mimic-iv/processed/fold_1`, not to a duplicated
`processed/processed/fold_1` directory.

- Subjects are split before episode rows are encoded. No subject may appear in
  more than one split.
- Normalization mean, standard deviation, and imputation median are computed
  only from training subjects.
- `train/val/test_raw_x.pkl` and missing masks are generated from their own
  split rather than the complete dataframe.
- `numerical_features.pkl` contains the 12 numerical features.
- `labtest_features.pkl` contains the complete one-hot dynamic feature order.
- Model input order is static `[Sex, Age]`, categorical one-hot columns, then
  numerical features.
- `basic.pkl` is keyed by the same pseudonymous `RecordID` stored in PID files.
- No notes, RAG output, DoctorAgent output, or expert prediction is generated.

## Local smoke result

- formatted episodes/rows: 600 / 28,800;
- labels: 80 outcome-positive and 520 outcome-negative;
- split: 540/30/30 subjects, with 72/4/4 positive labels;
- intersections: train-val 0, train-test 0, val-test 0;
- small-cohort dimensions: `demo_dim=2`, `lab_dim=41`, total input dimension 43;
- sequence shape: `x=[48,43]`, `y=[48,3]`, missing mask `[48,43]`.

The smoke data contain 29 observed categorical levels plus 12 numerical
features (`lab_dim=41`). The recovered historical contract contains 47 plus
12 (`lab_dim=59`). In the full MIMIC-IV 2.2 cohort, the same 29 levels are
observed and 18 historical compatibility columns remain all-zero. Thus the
18-column difference is categorical vocabulary/legacy encoding, not 18
additional physiologic variables.

## Formal local contract (2026-07-21)

The formal configuration is `configs/mimiciv_formal.json`; the smoke settings
remain separately recorded in `configs/mimiciv_smoke.json`.

| Rule | Evidence class | Formal choice |
|---|---|---|
| Outcome | user requirement and repository task order | `hospital_expire_flag`, 0 alive / 1 died in hospital |
| Split | paper and repository intent | stratified 90%/5%/5% by subject, seed 42 |
| Sample unit | repository input is a sequence; exact paper rule absent | first ICU stay per subject |
| First-stay ordering | reproduction choice | select the earliest ICU stay first, then apply eligibility |
| Age | reproduction/benchmark choice | adult, age 18 through 120 |
| Observation | historical preprocess keeps first 48 records; paper is silent | first 48 ICU hours, 1-hour bins |
| Short stays | required by the fixed observation window | exclude first ICU stays shorter than 48 hours |
| Sparse episodes | reproduction choice avoiding measurement-selection bias | retain; do not require a minimum observed-hour count |
| Categorical encoding | recovered repository history | fixed 47-column order, including inactive legacy columns |
| Normalization/imputation | repaired repository contract | training subjects only; forward fill then training median |

Cohort sensitivity on the small MIMIC tables, before reading events:

| Deterministic rule | Episodes | Outcome positive rate |
|---|---:|---:|
| all adult stays with LOS at least 48 h | 35,131 | 14.153% |
| first ICU per admission, then LOS filter | 30,983 | 13.291% |
| first eligible 48 h ICU per subject | 27,192 | 13.096% |
| first ICU per subject, then LOS filter (chosen) | 23,653 | 12.751% |

The chosen rule is not a count-fitting operation. It is the least ambiguous
one-episode-per-subject interpretation and prevents a later stay from replacing
an ineligible first stay. It yields 4,322 more samples than the paper's 19,331;
the paper does not publish enough cohort detail to justify deleting those
additional records.

## Formal variable sources

The 17 raw dynamic variables remain the five categorical and 12 numerical
features listed above. `feature_mapping.json` generated beside the formal
parquet records every item ID, range, unit conversion, and source policy.

- Vital signs, FiO2, height, weight, capillary refill, GCS components and the
  charted glucose/pH values come from selected `chartevents` item IDs.
- Temperature Fahrenheit is converted to Celsius; height inches to cm; weight
  pounds to kg; FiO2 percentages to fractions.
- Implausible numerical values are rejected using the configured clinical
  ranges before binning.
- GCS total uses a valid charted total when present, otherwise it is derived
  only when eye, motor, and verbal components all exist in that hour.
- To represent actual laboratory results rather than treating charted copies
  as equivalent, blood glucose item IDs 50809, 50931, 52027 and 52569 and blood
  pH item ID 50820 are also streamed from `labevents`. Other body fluids and
  urine are excluded. Within the same feature/hour, a valid labevent overrides
  a chartevent; within one source the latest timestamp wins.

Adding these five blood-lab item IDs is an explicit reproduction choice. The
checked-in ColaCare tree lacks its raw extractor, and the later public
PKU-AICare MIMIC preprocessor processes `chartevents` only. Therefore this
choice must remain fixed across all baselines and must not be described as a
paper-confirmed setting.

## Formal local execution result

- `FULL_SCAN=true` was logged throughout.
- chartevents: 313,645,063 rows, 10,248,411 valid target events seen, 413.915 s;
- labevents: 118,171,367 rows, 241,442 valid target events seen, 156.752 s;
- total raw rows streamed: 431,816,430;
- raw-to-parquet wall time: 620.254 s;
- cohort: 23,653 episodes/subjects, 3,016 positive and 20,637 negative;
- formatted parquet: 1,135,344 rows (48 per episode), 10,094,877 bytes;
- pickle splits: 21,287 / 1,183 / 1,183, with 2,714 / 151 / 151 positives;
- tensor contract: `x=[48,61]`, `y=[48,3]`, mask `[48,61]`;
- PatientID and RecordID intersections across all split pairs: zero;
- no NaN/inf in encoded `x`, `y`, or masks; labels match parquet;
- fixed dynamic dimension: 59; observed categorical levels: 29; inactive
  historical compatibility levels: 18.

The formal RETAIN plumbing run used one epoch, batch size 32, hidden size 128,
and CUDA on an RTX 4060 Laptop GPU. It completed in 10.780 s (8.267 s for the
epoch), wrote and reloaded its checkpoint, and returned finite probabilities.
The training loss 0.3277 and evaluation values are smoke diagnostics only, not
paper-comparable metrics.

## Formal-v2 correction (2026-07-29)

The formal-v1 fixed-vocabulary encoder converted missing categorical values to
strings before one-hot matching. This made all 843,807 missing GCS eye-opening
elements activate the historical `->None` column while their masks still
marked them missing. See `docs/MIMICIV_GCS_NONE_AUDIT.md` for the raw-table,
parquet, split, and mask evidence.

Formal-v2 is regenerated from the unchanged formatted parquet with the same
cohort, labels, subject split, normalization policy, feature order, and
59-dimensional dynamic contract. It differs only in correct categorical
missing-value handling. The strengthened validator reports zero one-hot values
at missing positions and 18 actual all-zero historical columns.

Consequences:

- formal-v1 data and model results are retained for audit only;
- RETAIN, ConCare, and AdaCare must all be retrained on formal-v2;
- no v1 checkpoint or expert output may be combined with v2 results;
- the previous AdaCare result is independently invalid because its recurrent
  layer interpreted `[B,T,F]` as `[T,B,F]`; see
  `docs/ADACARE_BATCH_SEMANTICS.md`.
