"""Exact score-tie uncertainty bounds for analyst review budgets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import _alert_budget_metrics, _alert_budget_tie_bounds, evaluate  # noqa: E402


def test_mixed_label_cutoff_tie_reports_exact_extremes_and_expectation():
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Two review slots: one certain positive and one of three tied rows,
    # of which one is positive and two are negative.
    bounds = _alert_budget_tie_bounds(labels, scores, fraction=0.2)
    assert bounds == pytest.approx((0.5, 1.0, 1 / 3, 2 / 3))
    expected_p, expected_r, budget = _alert_budget_metrics(labels, scores, fraction=0.2)
    assert budget == 2
    assert expected_p == pytest.approx(2 / 3)
    assert expected_r == pytest.approx(4 / 9)
    assert bounds[0] <= expected_p <= bounds[1]
    assert bounds[2] <= expected_r <= bounds[3]


def test_bounds_are_invariant_to_holdout_order_with_tied_scores():
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = _alert_budget_tie_bounds(labels, scores, fraction=0.2)
    for seed in range(8):
        permutation = np.random.default_rng(seed).permutation(len(labels))
        assert _alert_budget_tie_bounds(
            labels[permutation], scores[permutation], fraction=0.2
        ) == pytest.approx(expected)


def test_unique_scores_collapse_bounds_to_observed_top_budget():
    labels = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    scores = np.arange(10, dtype=float)
    expected_p, expected_r, _ = _alert_budget_metrics(labels, scores, fraction=0.3)
    lower_p, upper_p, lower_r, upper_r = _alert_budget_tie_bounds(
        labels, scores, fraction=0.3
    )
    assert (lower_p, upper_p) == pytest.approx((expected_p, expected_p))
    assert (lower_r, upper_r) == pytest.approx((expected_r, expected_r))


@pytest.mark.parametrize("bad_labels,bad_scores", [
    (np.array([0, 2, 1]), np.array([0.1, 0.2, 0.3])),
    (np.array([0, 1, 1]), np.array([0.1, np.nan, 0.3])),
    (np.array([0, 1, 1]), np.array([0.1, np.inf, 0.3])),
])
def test_tie_bounds_preserve_existing_metric_validation(bad_labels, bad_scores):
    with pytest.raises(ValueError):
        _alert_budget_tie_bounds(bad_labels, bad_scores, fraction=0.5)


def test_evaluation_includes_consistent_bounds_without_claiming_confidence():
    metrics = evaluate(n=400, seed=17)
    for metric in ("precision", "recall"):
        expected = metrics[f"{metric}_at_5pct_budget"]
        lower = metrics[f"{metric}_at_5pct_budget_lower_bound"]
        upper = metrics[f"{metric}_at_5pct_budget_upper_bound"]
        assert 0 <= lower <= expected + 1e-12
        assert expected <= upper + 1e-12 <= 1 + 1e-12
