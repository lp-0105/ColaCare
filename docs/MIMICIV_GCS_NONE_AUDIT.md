# MIMIC-IV GCS `None` contract audit

## Decision

The historical value `Glascow coma scale eye opening->None` is a legitimate
member of ColaCare's recovered 47-column categorical vocabulary, and literal
`None` values do occur in raw MIMIC-IV `chartevents`. However, its activation
in the formal-v1 pickle tensors is a preprocessing bug.

Every nonzero value in that formal-v1 tensor column came from a missing parquet
value converted by `Series.astype(str)` into the string `"None"`. The missing
mask simultaneously marked those same elements missing. No literal `None`
event survived into the selected formal cohort's formatted parquet.

This is therefore a data-contract bug (case B), not merely a documentation
error. Formal-v1 data and all checkpoints trained from it remain preserved but
must not be mixed with formal-v2 results.

## Evidence

- The recovered historical categorical contract contains
  `Glascow coma scale eye opening->None`.
- `d_items.csv.gz` identifies item 220739 as `GCS - Eye Opening` from
  `chartevents`.
- A complete chunked scan of 313,645,063 `chartevents` rows found 1,630,825
  rows for item 220739 and 202,266 literal raw values equal to `None`.
- The same scan found no empty/null `value` among those item rows.
- The selected formal parquet contains 1,135,344 hourly rows. Its GCS eye
  column contains 843,807 null values and zero literal `None` values.
- The formal-v1 pickle column was nonzero 758,967 / 42,474 / 42,366 times in
  train/validation/test. All 843,807 activations exactly overlapped elements
  marked missing in the mask.
- The source was `_encode_split` in
  `utils/mimiciv_preprocess_utils.py`: `source.astype(str)` converted Python
  missing values to the historical string level before one-hot comparison.

No patient, admission, stay, record identifier, or raw patient row was emitted
during the audit.

## Correction

Missing categorical values are filled with the empty string before conversion
to text. The fixed 59-dimensional contract remains unchanged:

- 2 demographic inputs;
- 47 fixed categorical one-hot inputs;
- 12 numerical inputs.

The historical `->None` column remains present but is all-zero for this cohort.
Formal-v2 is regenerated from the unchanged formatted parquet, so the
431,816,430-row raw event scan is not repeated and cohort/label/split semantics
do not change.

The strengthened validator now compares all categorical tensor counts with the
formatted parquet and rejects any nonzero one-hot value whose missing mask is
active. Formal-v2 passes with:

- 23,653 episodes;
- split 21,287 / 1,183 / 1,183;
- positives 2,714 / 151 / 151;
- shapes `x=(48,61)`, `y=(48,3)`, `mask=(48,61)`;
- zero patient or record overlap;
- zero missing-one-hot violations;
- 18 actual all-zero historical compatibility columns.

Because encoded inputs changed, RETAIN, ConCare, and AdaCare must all be
retrained from random initialization on formal-v2.
