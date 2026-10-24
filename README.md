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
standardized before the model sees them (`src/detector.py`). Input must be a
nonempty 2D numeric array with exactly six columns in the listed order; all
measurements must be finite and nonnegative. Invalid telemetry is rejected
rather than silently converted (in particular, negative byte counts are **not**
turned positive with an absolute-value transform). This validation runs for
both fitting and inference.

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

On the bundled synthetic dataset (4000 flows, 5% anomalies), evaluation now
uses a stratified **75% train / 25% held-out test split**. The scaler and
Isolation Forest are fit only on training flows; labels are used only for
stratification and final scoring. This prevents the portfolio benchmark from
quietly evaluating on records the detector already saw during fitting.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Tests cover dataset generation, binary predictions, anomaly score ordering,
an obvious DoS-shaped flow, detection quality, held-out evaluation integrity,
and malformed feature rejection during training and inference.

## ⚠️ Honesty about the data

The dataset (`src/data.py`) is **synthetic**, with anomalies drawn from
deliberately distinct attack-like distributions — so the model scores very
high here. Real traffic (CICIDS, UNSW-NB15) is messier and the numbers would be
lower; labeled captures are large and licensed, hence the synthetic generator
for a reproducible offline pipeline. The feature extraction and model code are
unchanged if you point them at a real flow CSV.

## Author

Koushik Chowdary — [LinkedIn](https://linkedin.com/in/koushik-chowdary) · [GitHub](https://github.com/koushikchowdary6)
