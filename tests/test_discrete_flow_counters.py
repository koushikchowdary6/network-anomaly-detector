"""Synthetic NetFlow counters must be physically discrete and internally consistent."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import generate


@pytest.mark.parametrize("contamination", [0.0, 0.05, 0.5])
def test_bytes_and_packets_are_integer_valued_for_every_flow(contamination):
    X, labels = generate(n=600, contamination=contamination, seed=23)
    assert X.shape == (600, 6)
    assert len(labels) == 600
    np.testing.assert_array_equal(X[:, 1:5], np.rint(X[:, 1:5]))
    assert np.all(X[:, 1:5] >= 0)
    assert np.all(X[:, 3] >= 1)
    np.testing.assert_allclose(
        X[:, 5], X[:, 3] / np.maximum(X[:, 0], 0.01), rtol=1e-12
    )


def test_discrete_counter_generation_is_seed_reproducible():
    first, labels = generate(n=500, contamination=0.12, seed=99)
    second, repeated_labels = generate(n=500, contamination=0.12, seed=99)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(labels, repeated_labels)
