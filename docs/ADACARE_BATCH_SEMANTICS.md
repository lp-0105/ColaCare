# AdaCare batch/time semantics correction

## Root cause

ColaCare passes AdaCare input as `[batch, time, feature]`. The convolution and
recalibration path preserves that semantic layout and constructs recurrent
input with shape `[batch, time, feature + 3 * kernel_num]`.

The AdaCare GRU/LSTM was created with PyTorch's default
`batch_first=False`. It therefore interpreted the first dimension as time and
the second as batch. Its output happened to retain the same three numeric
dimensions, allowing downstream `get_last_visit` indexing to run without a
shape exception while samples in a minibatch influenced one another.

The previous AdaCare checkpoint reproduced its old predictions only under the
original sample order and batch size. The measured diagnostic errors were:

- shuffled batch then restored: maximum probability error 0.113173;
- fixed samples evaluated individually: maximum probability error 0.168737.

Those checkpoint metrics are invalid as patient-independent predictions and
must be labeled `INVALID_BATCH_SEMANTICS_V1`.

## Minimal correction

The model retains the repository's `[B,T,F]` design. Both AdaCare recurrent
variants now set `batch_first=True`; no feature definition, hidden dimension,
loss, convolution, attention, or output head is changed.

The resulting semantic path is:

1. input `[B,T,F]`;
2. causal convolutions use `[B,F,T]`;
3. recalibrated values return to `[B,T,*]`;
4. GRU/LSTM consumes and returns `[B,T,H]`;
5. `get_last_visit` selects a time step per patient;
6. patient embedding is `[B,H]`;
7. native input importance is `[B,T,F]`.

## Regression gates

`tests/test_adacare_semantics.py` covers:

- batch permutation invariance;
- one sample alone and with two different companion groups;
- batch sizes 1, 2, 8, 32, and 128;
- checkpoint round-trip;
- `B=1` and `B>1` shapes for logits, probability, embedding, and importance;
- finite forward outputs and finite backward gradients;
- a successful optimizer step.

The prediction tolerance is `1e-5`; checkpoint reload tolerance is `1e-7`.
Formal L20 training may begin only after these tests and the equivalent
formal-data runtime gate pass.

On the L20, the corrected model initially showed batch-size-only numerical
differences up to `6.89e-05` while PyTorch allowed TF32. Disabling TF32 for
CUDA matrix multiplication and cuDNN reduced the maximum error to
`5.96e-08`, with zero permutation and companion-sample error. The formal
expert-stage runner therefore fixes matrix multiplication to strict FP32 in
`seed_everything`; this is a numerical reproducibility setting and does not
change the AdaCare architecture or feature contract.
