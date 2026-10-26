"""Evaluate on held-out flows, never the records used to fit the detector."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from data import FEATURES, generate
from detector import AnomalyDetector



def _alert_budget_metrics(
    y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.05
) -> tuple[float, float, int]:
    """Rank holdout flows by anomaly score for a fixed analyst review budget.

    This evaluates score ranking, independently of the model's binary
    contamination threshold. Labels are used only after scoring.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("alert budget fraction must be in (0, 1]")
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores) or not len(y_true):
        raise ValueError("alert budget inputs must be aligned non-empty vectors")
    positives = int(np.sum(y_true))
    if positives == 0 or not np.isfinite(scores).all():
        raise ValueError("alert budget requires positives and finite scores")
    budget = max(1, math.ceil(len(y_true) * fraction))
    # Stable sorting makes tied-score handling reproducible across runs.
    top_indices = np.argsort(-scores, kind="stable")[:budget]
    hits = int(np.sum(y_true[top_indices]))
    return hits / budget, hits / positives, budget


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
    budget_precision, budget_recall, budget_count = _alert_budget_metrics(y_test, scores)
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "test_prevalence": float(np.mean(y_test)),
        "precision_at_5pct_budget": budget_precision,
        "recall_at_5pct_budget": budget_recall,
        "alerts_at_5pct_budget": budget_count,
        "precision": float(p), "recall": float(r), "f1": float(f),
        "n": n, "anomalies": int(y.sum()),
        "train_flows": len(X_train), "test_flows": len(X_test),
        "test_anomalies": int(y_test.sum()),
    }


if __name__ == "__main__":
    m = evaluate()
    print(f"Isolation Forest on {m['n']} flows ({m['anomalies']} anomalies, unsupervised):")
    print(f"  ROC-AUC  : {m['roc_auc']:.3f}")
    print(f"  PR-AUC (average precision): {m['average_precision']:.3f}")
    print(f"  test prevalence: {m['test_prevalence']:.3f}")
    print(f"  precision @ 5% alert budget: {m['precision_at_5pct_budget']:.3f}")
    print(f"  recall @ 5% alert budget: {m['recall_at_5pct_budget']:.3f}")
    print(f"  precision: {m['precision']:.3f}")
    print(f"  recall   : {m['recall']:.3f}")
    print(f"  f1       : {m['f1']:.3f}")
