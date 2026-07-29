from __future__ import annotations

import numpy as np

from scripts.run_mimiciv_expert_stage import (
    aggregate_shap_values,
    select_anonymous_test_indices,
)


def test_anonymous_sample_selection_is_deterministic_and_contains_both_labels():
    labels = np.array([0] * 40 + [1] * 12, dtype=np.int64)

    first = select_anonymous_test_indices(labels, sample_count=32, seed=42)
    second = select_anonymous_test_indices(labels, sample_count=32, seed=42)

    assert np.array_equal(first, second)
    assert len(first) == 32
    assert len(np.unique(first)) == 32
    assert set(labels[first].tolist()) == {0, 1}


def test_repository_shap_aggregation_preserves_last_bin_feature_semantics():
    values = np.arange(3 * 61, dtype=np.float64).reshape(3, 61) - 50

    aggregation = aggregate_shap_values(values)

    assert aggregation["sample_feature_abs"].shape == (3, 61)
    assert aggregation["sample_abs_sum"].shape == (3,)
    assert aggregation["feature_abs_mean"].shape == (61,)
    assert aggregation["sample_time_abs"] is None
    assert aggregation["time_semantics"] == "last_observation_only"
