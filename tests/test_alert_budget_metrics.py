"""Operational ranking metrics for unsupervised anomaly scores."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import _alert_budget_metrics, evaluate


def test_alert_budget_uses_highest_scores_not_classifier_labels():
    labels = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.1, 0.95, 0.2, 0.85, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    precision, recall, count = _alert_budget_metrics(labels, scores, fraction=0.2)
    assert (precision, recall, count) == (1.0, 1.0, 2)


def test_alert_budget_rounds_up_and_shares_boundary_ties():
    labels = np.array([0, 1, 1])
    scores = np.array([0.9, 0.9, 0.1])
    # One of two equally ranked candidates is positive. With one review
    # slot, expected precision is 1/2 and expected recall is 1/4.
    assert _alert_budget_metrics(labels, scores) == (0.5, 0.25, 1)
    assert _alert_budget_metrics(labels, scores, fraction=2 / 3) == (0.5, 0.5, 2)


def test_alert_budget_boundary_ties_are_invariant_to_holdout_row_order():
    labels = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.95, 0.8, 0.8, 0.8, 0.1])
    # Two review slots: one guaranteed positive above the boundary and
    # one randomly selected from three tied rows with one positive.
    expected = (2 / 3, 4 / 9, 2)
    for permutation in (
        np.array([0, 1, 2, 3, 4]),
        np.array([3, 2, 4, 0, 1]),
        np.array([4, 1, 0, 3, 2]),
    ):
        precision, recall, budget = _alert_budget_metrics(
            labels[permutation], scores[permutation], fraction=0.4
        )
        assert precision == pytest.approx(expected[0])
        assert recall == pytest.approx(expected[1])
        assert budget == expected[2]


def test_alert_budget_without_boundary_ties_keeps_exact_hit_counts():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert _alert_budget_metrics(labels, scores, fraction=0.5) == (0.5, 0.5, 2)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_alert_budget_fails_explicitly(fraction):
    with pytest.raises(ValueError, match="alert budget fraction"):
        _alert_budget_metrics(np.array([0, 1]), np.array([0.1, 0.9]), fraction)


def test_mismatched_vectors_and_nonfinite_scores_fail_explicitly():
    with pytest.raises(ValueError, match="aligned"):
        _alert_budget_metrics(np.array([0, 1]), np.array([0.1]))
    with pytest.raises(ValueError, match="finite"):
        _alert_budget_metrics(np.array([0, 1]), np.array([0.1, np.nan]))


def test_heldout_evaluation_reports_operational_ranking_metrics():
    result = evaluate(n=2000, contamination=0.05, seed=3)
    assert 0 <= result["average_precision"] <= 1 + 1e-12  # floating-point PR-AUC roundoff
    assert result["average_precision"] > result["test_prevalence"]
    assert result["alerts_at_5pct_budget"] == 25
    assert 0 <= result["precision_at_5pct_budget"] <= 1
    assert 0 <= result["recall_at_5pct_budget"] <= 1
    assert result["test_prevalence"] == pytest.approx(
        result["test_anomalies"] / result["test_flows"]
    )


@pytest.mark.parametrize("labels", [
    np.array([0, 2, 1]),
    np.array([0, -1, 1]),
    np.array([0, 0.5, 1]),
    np.array([0, np.nan, 1]),
    np.array(["0", "1", "1"]),
])
def test_nonbinary_ground_truth_cannot_inflate_alert_metrics(labels):
    with pytest.raises(ValueError, match="binary"):
        _alert_budget_metrics(labels, np.array([0.1, 0.9, 0.8]), fraction=1)


@pytest.mark.parametrize("scores", [
    np.array([0.1, np.inf]),
    np.array([0.1, complex(0.9, 1)]),
    np.array(["low", "high"]),
])
def test_nonreal_or_nonnumeric_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="finite real-valued scores"):
        _alert_budget_metrics(np.array([0, 1]), scores)


def test_valid_binary_alert_budget_is_bounded():
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    precision, recall, count = _alert_budget_metrics(labels, scores, fraction=0.5)
    assert (precision, recall, count) == (1.0, 1.0, 2)
