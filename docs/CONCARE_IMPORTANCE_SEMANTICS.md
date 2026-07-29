# ConCare 60-dimensional importance semantics

## Confirmed conclusion

The exported ConCare importance vector has exactly 60 positions:

- positions `0..58`: the 59 formal_v2 dynamic input channels, in the exact
  order of `labtest_features.pkl`;
- position `59`: one joint demographic context obtained by projecting the two
  static inputs, Sex and Age, together.

It is **not** a 60th laboratory variable, does not exclude an unexplained
dynamic feature, and cannot be expanded into separate Sex and Age weights.
The complete machine-readable mapping is
`configs/concare_importance_mapping_v1.json`.

## Code evidence and dimension flow

1. `scripts/run_mimiciv_expert_stage.py::OutcomeModel.forward` calls ConCare
   with:

   - dynamic input `x[:, :, 2:]`, shape `[B,48,59]`;
   - static input `x[:,0,:2]`, shape `[B,2]`;
   - a sequence-validity mask of shape `[B,48]` containing all ones.

2. `pyehr/models/concare.py::ConCareLayer` creates one independent GRU and one
   temporal `SingleAttention` module for every dynamic channel. This reduces
   each feature's 48-hour sequence to one hidden vector, preserving the
   dynamic feature order.

3. The two demographic values are jointly projected by one linear layer,
   `demo_proj_main`, producing one additional hidden vector.

4. The model concatenates dynamic contexts first and `demo_main` last. The
   resulting token order is therefore 59 dynamic tokens followed by one
   demographics token.

5. Multi-head attention contextualises these 60 tokens. `FinalAttentionQKV`
   applies a softmax across the token dimension and returns `feature_attn`.
   `scripts/run_mimiciv_expert_stage.py::predict_in_batches` saves this tensor
   directly as `feature_importance`, yielding `[N,60]`.

## What the values mean

The values are final cross-channel attention weights over already
time-aggregated, contextualised channel representations.

- They are non-negative softmax weights in evaluation mode and sum to one
  subject to ordinary floating-point tolerance.
- They are unsigned attention, not SHAP values, gradients, causal effects, or
  signed contributions to the outcome logit.
- They can rank channels within one ConCare output.
- Their magnitude must not be compared directly with RETAIN or AdaCare native
  importance values.
- Because multi-head attention contextualises each channel before the final
  attention, a high weight means the final ConCare pooling attends to that
  contextualised channel; it does not prove that the raw feature alone caused
  the prediction.

## Time semantics

The 48-hour axis has already been reduced separately for each dynamic feature
by `SingleAttention`. The exported tensor does not include the per-feature
temporal attention arrays. Therefore:

- no exact hour can be recovered from `[60]`;
- the adapter must emit `time_hour=null`;
- it must not attach a latest, maximum, or otherwise invented raw value to the
  attention weight;
- RETAIN/AdaCare `[48,61]` evidence and ConCare `[60]` evidence require
  different rendering paths.

## Static demographics

Position 59 represents `tanh(demo_proj_main([Sex, Age]))` as one token. Sex and
Age are not separately identifiable in this importance vector. The adapter
names this position `Demographics context (Sex + Age)`, assigns no unit or raw
value, and never pretends it is a clinical measurement.

## Mask and missingness semantics

The mask passed to ConCare in the formal expert runner is a sequence-validity
mask with all 48 hours marked valid. It is not the formal_v2 61-column
measurement missing mask.

ConCare therefore operates on the normalised/imputed dynamic input and does
not receive missingness as an extra clinical feature. The adapter independently
uses `raw_x` plus `missing_mask` to describe whether a mapped feature was
observed anywhere in the window. That status is context about the source data,
not part of the exported attention calculation.

## The 18 inactive compatibility columns

formal_v2 preserves 47 historical one-hot dimensions, of which 18 are
inactive under the corrected category contract. They retain their ordinary
positions among ConCare indices `0..46`; the machine-readable mapping lists
their names.

An inactive raw input channel is not guaranteed to have exactly zero final
ConCare attention. Biases and cross-channel contextualisation can produce a
non-zero attention weight even when that raw channel is always zero. The
adapter therefore marks their contract status but does not fabricate a zero
weight or discard them.

## Mapping safety checks

The adapter refuses ConCare evidence unless all of the following hold:

- importance shape is exactly `[60]`;
- all values are finite;
- values are non-negative and sum to one within floating-point tolerance;
- mapping indices are exactly `0..59`;
- mapping positions `0..58` exactly equal the formal_v2 dynamic feature order;
- mapping position 59 explicitly sources input indices `[0,1]`;
- the declared semantics match the audited unsigned, time-aggregated contract.

No zero-padding or guessed 61st position is allowed.
