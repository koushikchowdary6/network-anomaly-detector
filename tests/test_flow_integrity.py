"""Flow telemetry consistency and failure-safe model retraining contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import generate
from detector import AnomalyDetector
from evaluate import evaluate


def test_synthetic_rate_is_derived_from_packets_and_duration_for_both_classes():
    X, labels = generate(n=600, contamination=0.1, seed=17)
    assert set(labels) == {0, 1}
    np.testing.assert_allclose(
        X[:, 5], X[:, 3] / np.maximum(X[:, 0], 0.01), rtol=1e-12
    )
    X_again, labels_again = generate(n=600, contamination=0.1, seed=17)
    np.testing.assert_array_equal(X, X_again)
    np.testing.assert_array_equal(labels, labels_again)


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "100"])
def test_invalid_flow_counts_fail_explicitly(n):
    with pytest.raises(ValueError, match="positive integer"):
        generate(n=n)


@pytest.mark.parametrize("contamination", [
    -0.01, 0.51, float("nan"), float("inf"), True, "0.05",
])
def test_invalid_contamination_fails_explicitly(contamination):
    with pytest.raises(ValueError, match="contamination"):
        generate(contamination=contamination)


@pytest.mark.parametrize(("n", "contamination"), [
    (20, 0.05),  # only one anomaly: stratification is impossible
    (10, 0.0),  # ROC-AUC is undefined for a single class
])
def test_evaluation_rejects_unscorable_holdout(n, contamination):
    with pytest.raises(ValueError, match="held-out evaluation"):
        evaluate(n=n, contamination=contamination)


def test_failed_refit_preserves_previous_scaler_and_forest(monkeypatch):
    X, _ = generate(n=250, seed=9)
    detector = AnomalyDetector(seed=9).fit(X)
    before_scores = detector.anomaly_score(X[:12])
    before_predictions = detector.predict(X[:12])
    original_scaler, original_model = detector.scaler, detector.model

    def fail_fit(self, X, y=None, **kwargs):
        raise RuntimeError("simulated forest fit failure")

    monkeypatch.setattr(IsolationForest, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="simulated forest fit failure"):
        detector.fit(X * 1.5)

    assert detector.scaler is original_scaler
    assert detector.model is original_model
    np.testing.assert_array_equal(detector.anomaly_score(X[:12]), before_scores)
    np.testing.assert_array_equal(detector.predict(X[:12]), before_predictions)
