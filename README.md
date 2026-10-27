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

The evaluation also reports **average precision (PR-AUC)** and the held-out
positive-class prevalence (the random-ranking baseline), since ROC-AUC alone
can obscure alert quality in imbalanced detection. It ranks held-out flows
by anomaly score and reports **precision/recall among the top 5%** that an
analyst could review, with the alert count rounded up. This fixed review
budget is independent of the Isolation Forest's binary contamination
threshold; labels never enter training or score ranking. When the alert-budget
cutoff falls within a tied-score group, precision and recall use expected
fractional credit under a uniform random tie-break rather than arbitrarily
favoring earlier holdout rows. The alert count remains an integer, while the
reported hit count is an expectation, not an observed deployment outcome.
The top-5% figures are synthetic benchmark measurements, not production SOC performance.
The alert-budget evaluator rejects nonbinary ground-truth labels and
nonfinite, nonreal, or nonnumeric scores before ranking. This prevents
malformed labels (such as a value of 2) from inflating hit counts into
impossible precision/recall values above 100%.

Synthetic attack-like flows now compute packets-per-second from their packet
counts and duration, rather than independently sampling an impossible rate.
Generator inputs are validated; evaluation fails explicitly if too few
examples exist for a two-class held-out benchmark. A failed model refit keeps
the previous scaler and forest together, avoiding corrupted inference state.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Tests cover dataset generation, binary predictions, anomaly score ordering,
an obvious DoS-shaped flow, detection quality, held-out evaluation integrity,
malformed feature rejection, synthetic feature consistency, insufficient
holdout classes, atomic refit behavior, and top-score alert-budget ranking
(including ties, rounding, and invalid metric inputs).

## ⚠️ Honesty about the data

The dataset (`src/data.py`) is **synthetic**, with anomalies drawn from
deliberately distinct attack-like distributions — so the model scores very
high here. Real traffic (CICIDS, UNSW-NB15) is messier and the numbers would be
lower; labeled captures are large and licensed, hence the synthetic generator
for a reproducible offline pipeline. The feature extraction and model code are
unchanged if you point them at a real flow CSV.

## Author

Koushik Chowdary — [LinkedIn](https://linkedin.com/in/koushik-chowdary) · [GitHub](https://github.com/koushikchowdary6)
