"""Evaluate on held-out flows, never the records used to fit the detector."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from data import FEATURES, generate
from detector import AnomalyDetector


def evaluate(n: int = 4000, contamination: float = 0.05, seed: int = 42):
    X, y = generate(n, contamination, seed)
    # Stratified evaluation and ROC-AUC require both classes in the holdout.
    # Surface insufficient synthetic data as an explicit contract error.
    counts = np.bincount(y, minlength=2)
    if min(counts) < 2 or int(np.ceil(len(y) * 0.25)) < 2:
        raise ValueError(
            "held-out evaluation requires at least two normal and two "
            "anomalous flows and two test records"
        )
    # Keep the evaluation partition completely out of model/scaler fitting.
    # Labels are used for stratification and evaluation only, not for fitting.
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    det = AnomalyDetector(contamination=contamination, seed=seed).fit(X_train)
    preds = det.predict(X_test)
    scores = det.anomaly_score(X_test)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "n": n, "anomalies": int(y.sum()),
        "train_flows": len(X_train), "test_flows": len(X_test),
        "test_anomalies": int(y_test.sum()),
    }


if __name__ == "__main__":
    m = evaluate()
    print(f"Isolation Forest on {m['n']} flows ({m['anomalies']} anomalies, unsupervised):")
    print(f"  ROC-AUC  : {m['roc_auc']:.3f}")
    print(f"  precision: {m['precision']:.3f}")
    print(f"  recall   : {m['recall']:.3f}")
    print(f"  f1       : {m['f1']:.3f}")
