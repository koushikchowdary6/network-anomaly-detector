"""Regression tests for fail-closed network-flow feature preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from data import FEATURES, generate
from detector import AnomalyDetector


@pytest.mark.parametrize(("bad", "message"), [
    (np.full((1, len(FEATURES)), -1.0), "nonnegative"),
    (np.array([[0, 1, 2, -3, 4, 5]], dtype=float), "nonnegative"),
    (np.array([[0, 1, np.nan, 3, 4, 5]]), "finite"),
    (np.array([[0, 1, np.inf, 3, 4, 5]]), "finite"),
    (np.zeros((2, len(FEATURES) - 1)), "shape"),
    (np.zeros(len(FEATURES)), "shape"),
    (np.empty((0, len(FEATURES))), "at least one"),
    (np.array([["not-a-number"] * len(FEATURES)]), "numeric"),
])
def test_fit_rejects_invalid_telemetry(bad, message):
    with pytest.raises(ValueError, match=message):
        AnomalyDetector().fit(bad)


def test_invalid_refit_does_not_destroy_previous_fitted_state():
    X, _ = generate(200, seed=7)
    detector = AnomalyDetector(seed=7).fit(X)
    before = detector.anomaly_score(X[:10])
    invalid = X.copy()
    invalid[0, 1] = -100
    with pytest.raises(ValueError, match="nonnegative"):
        detector.fit(invalid)
    np.testing.assert_array_equal(detector.anomaly_score(X[:10]), before)


@pytest.mark.parametrize("method", ["predict", "anomaly_score"])
def test_inference_rejects_invalid_flow_instead_of_mirroring_positive_values(method):
    X, _ = generate(200, seed=9)
    detector = AnomalyDetector(seed=9).fit(X)
    invalid = X[:1].copy()
    invalid[0, 1] = -invalid[0, 1]
    with pytest.raises(ValueError, match="nonnegative"):
        getattr(detector, method)(invalid)


def test_zero_counts_remain_valid():
    X, _ = generate(200, seed=10)
    detector = AnomalyDetector(seed=10).fit(X)
    zero_flow = np.zeros((1, len(FEATURES)))
    assert detector.predict(zero_flow).shape == (1,)
    assert np.isfinite(detector.anomaly_score(zero_flow)).all()
