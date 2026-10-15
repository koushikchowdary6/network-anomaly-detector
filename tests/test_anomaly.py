import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from data import generate, FEATURES
from detector import AnomalyDetector
from evaluate import evaluate


def test_dataset_shape_and_labels():
    X, y = generate(1000, contamination=0.05, seed=1)
    assert X.shape == (1000, len(FEATURES))
    assert int(y.sum()) == 50


def test_predict_returns_binary():
    X, _ = generate(500, seed=2)
    preds = AnomalyDetector().fit(X).predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_detects_injected_anomalies():
    m = evaluate(n=2000, contamination=0.05, seed=3)
    # Isolation Forest should recover the deliberately-distinct anomalies well.
    assert m["roc_auc"] > 0.9
    assert m["recall"] > 0.7


def test_scores_rank_anomalies_higher():
    X, y = generate(2000, seed=4)
    det = AnomalyDetector().fit(X)
    scores = det.anomaly_score(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_obvious_dos_flow_flagged():
    X, _ = generate(2000, seed=5)
    det = AnomalyDetector().fit(X)
    # A synthetic DoS-like flow: huge packet counts, high pps.
    dos = np.array([[2.0, 8000.0, 300.0, 15000.0, 30.0, 7000.0]])
    assert det.predict(dos)[0] == 1


def test_evaluation_keeps_holdout_out_of_training():
    m = evaluate(n=400, contamination=0.05, seed=11)
    assert m["train_flows"] == 300
    assert m["test_flows"] == 100
    assert m["train_flows"] + m["test_flows"] == m["n"]
    assert 0 < m["test_anomalies"] < m["anomalies"]
