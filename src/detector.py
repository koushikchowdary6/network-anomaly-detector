"""Unsupervised network anomaly detector using Isolation Forest."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data import FEATURES


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        )

    @staticmethod
    def _prepare_features(X: np.ndarray) -> np.ndarray:
        """Validate NetFlow measurements before any model or scaler operation.

        Absolute-value preprocessing would silently turn invalid negative
        byte/packet counts into plausible positive traffic. Reject those
        records instead of training or scoring on corrupted telemetry.
        """
        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("flow features must be numeric") from exc
        if values.ndim != 2 or values.shape[1] != len(FEATURES):
            raise ValueError(f"flow features must have shape (n, {len(FEATURES)})")
        if not values.shape[0]:
            raise ValueError("flow features must contain at least one record")
        if not np.isfinite(values).all():
            raise ValueError("flow features must be finite")
        if (values < 0).any():
            raise ValueError("flow features must be nonnegative")
        return np.log1p(values)

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        # A failed refit must not pair an old forest with a newly fitted scaler.
        # Train candidate components and publish them only after both succeed.
        prepared = self._prepare_features(X)
        candidate_scaler = StandardScaler()
        Xs = candidate_scaler.fit_transform(prepared)
        candidate_model = clone(self.model)
        candidate_model.fit(Xs)
        self.scaler = candidate_scaler
        self.model = candidate_model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal."""
        Xs = self.scaler.transform(self._prepare_features(X))
        return (self.model.predict(Xs) == -1).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        Xs = self.scaler.transform(self._prepare_features(X))
        return -self.model.score_samples(Xs)
