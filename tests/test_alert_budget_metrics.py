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


def test_alert_budget_rounds_up_and_handles_stable_ties():
    labels = np.array([0, 1, 1])
    scores = np.array([0.9, 0.9, 0.1])
    assert _alert_budget_metrics(labels, scores) == (0.0, 0.0, 1)
    assert _alert_budget_metrics(labels, scores, fraction=2 / 3) == (0.5, 0.5, 2)


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
    assert 0 <= result["average_precision"] <= 1
    assert result["average_precision"] > result["test_prevalence"]
    assert result["alerts_at_5pct_budget"] == 25
    assert 0 <= result["precision_at_5pct_budget"] <= 1
    assert 0 <= result["recall_at_5pct_budget"] <= 1
    assert result["test_prevalence"] == pytest.approx(
        result["test_anomalies"] / result["test_flows"]
    )
