"""Unsupervised network anomaly detector using Isolation Forest."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        )

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        # Log-scale heavy-tailed byte/packet counts before standardizing.
        Xs = self.scaler.fit_transform(np.log1p(np.abs(X)))
        self.model.fit(Xs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal."""
        Xs = self.scaler.transform(np.log1p(np.abs(X)))
        return (self.model.predict(Xs) == -1).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        Xs = self.scaler.transform(np.log1p(np.abs(X)))
        return -self.model.score_samples(Xs)
