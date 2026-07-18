"""Fit the detector on unlabeled traffic, then evaluate against held-out labels."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from data import FEATURES, generate
from detector import AnomalyDetector


def evaluate(n: int = 4000, contamination: float = 0.05, seed: int = 42):
    X, y = generate(n, contamination, seed)
    det = AnomalyDetector(contamination=contamination, seed=seed).fit(X)  # unsupervised
    preds = det.predict(X)
    scores = det.anomaly_score(X)
    p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "n": n, "anomalies": int(y.sum()),
    }


if __name__ == "__main__":
    m = evaluate()
    print(f"Isolation Forest on {m['n']} flows ({m['anomalies']} anomalies, unsupervised):")
    print(f"  ROC-AUC  : {m['roc_auc']:.3f}")
    print(f"  precision: {m['precision']:.3f}")
    print(f"  recall   : {m['recall']:.3f}")
    print(f"  f1       : {m['f1']:.3f}")
