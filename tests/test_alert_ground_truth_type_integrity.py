"""Alert-budget metrics must not silently coerce ground-truth label types."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import _alert_budget_metrics


@pytest.mark.parametrize("labels", [
    np.array([False, True]),
    np.array([0.0, 1.0]),
    np.array([0, 1], dtype=object),
    np.array([0.0, 1.0], dtype=np.float32),
])
def test_coerced_binary_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="binary integer"):
        _alert_budget_metrics(labels, np.array([0.1, 0.9]))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
def test_explicit_integer_labels_remain_supported(dtype):
    labels = np.array([0, 1, 0, 1], dtype=dtype)
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert _alert_budget_metrics(labels, scores, fraction=0.5) == (1.0, 1.0, 2)
