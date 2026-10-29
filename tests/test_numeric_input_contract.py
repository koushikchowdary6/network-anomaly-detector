"""Prevent silent type coercion in network telemetry and alert budgets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector import AnomalyDetector
from evaluate import _alert_budget_metrics


def _row(value):
    # Object dtype preserves booleans in mixed numeric inputs.
    return np.array([[1.0, 10.0, 5.0, value, 2.0, 3.0]], dtype=object)


@pytest.mark.parametrize("bad", [True, np.bool_(False), "12", "1.5", None, 1 + 2j])
def test_flow_features_reject_non_numeric_or_coerced_scalars(bad):
    with pytest.raises(ValueError, match="real numbers"):
        AnomalyDetector._prepare_features(_row(bad))


def test_nested_python_boolean_cannot_become_a_packet_count():
    with pytest.raises(ValueError, match="real numbers"):
        AnomalyDetector._prepare_features([[1, 10, 5, True, 2, 3]])


def test_real_numeric_input_still_transforms_and_rejects_invalid_values():
    valid = [[1, 10, 5, 3, 2, 3], [2.0, 12.0, 7.0, 4.0, 2.0, 2.0]]
    assert np.allclose(AnomalyDetector._prepare_features(valid), np.log1p(valid))
    assert np.allclose(
        AnomalyDetector._prepare_features(np.array(valid, dtype=object)),
        np.log1p(valid),
    )
    with pytest.raises(ValueError, match="finite"):
        AnomalyDetector._prepare_features(_row(np.inf))
    with pytest.raises(ValueError, match="nonnegative"):
        AnomalyDetector._prepare_features(_row(-1))


@pytest.mark.parametrize("fraction", [True, False, np.bool_(True), "0.5", 1 + 0j, None])
def test_alert_budget_rejects_non_real_fractions(fraction):
    with pytest.raises(ValueError, match="alert budget fraction"):
        _alert_budget_metrics(
            np.array([0, 1]), np.array([0.1, 0.9]), fraction=fraction
        )


def test_alert_budget_accepts_real_numpy_fraction():
    assert _alert_budget_metrics(
        np.array([0, 1]), np.array([0.1, 0.9]), fraction=np.float64(0.5)
    ) == (1.0, 1.0, 1)
