# Network Anomaly Detector 📡🌲

[![CI](https://github.com/koushikchowdary6/network-anomaly-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/koushikchowdary6/network-anomaly-detector/actions/workflows/ci.yml)

**Unsupervised** anomaly detection for network traffic using an **Isolation
Forest**. It learns the shape of normal NetFlow-style traffic and flags
outliers — port-scan bursts, data exfiltration, DoS floods — **without ever
being told which flows are attacks**. That's the point of unsupervised
detection for a SOC: it can surface novel attacks that a signature or a
supervised model trained on old labels would miss.

## Features per flow

`duration`, `src_bytes`, `dst_bytes`, `src_pkts`, `dst_pkts`, `pkts_per_sec` —
standard NetFlow fields. Heavy-tailed byte/packet counts are log-scaled and
standardized before the model sees them (`src/detector.py`).

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=src python src/evaluate.py
```

```python
from data import generate
from detector import AnomalyDetector
X, y = generate()                      # y is only for scoring, not training
det = AnomalyDetector().fit(X)         # unsupervised
det.predict(X)                         # 1 = anomaly, 0 = normal
det.anomaly_score(X)                   # higher = more anomalous
```

## Results

On the bundled synthetic dataset (4000 flows, 5% anomalies), the detector
recovers the injected attacks with high ROC-AUC and F1 — while never seeing the
labels during training.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

5 tests cover dataset generation, binary predictions, that anomalies score
higher than normal traffic, and that an obvious DoS-shaped flow is flagged.

## ⚠️ Honesty about the data

The dataset (`src/data.py`) is **synthetic**, with anomalies drawn from
deliberately distinct attack-like distributions — so the model scores very
high here. Real traffic (CICIDS, UNSW-NB15) is messier and the numbers would be
lower; labeled captures are large and licensed, hence the synthetic generator
for a reproducible offline pipeline. The feature extraction and model code are
unchanged if you point them at a real flow CSV.

## Author

Koushik Chowdary — [LinkedIn](https://linkedin.com/in/koushik-chowdary) · [GitHub](https://github.com/koushikchowdary6)
