"""Synthetic network-flow dataset with injected anomalies.

Generates NetFlow-style records: duration, bytes/packets in each direction,
packets-per-second, and a categorical protocol. ~95% "normal" traffic drawn
from realistic distributions, plus ~5% anomalies of a few attack-like shapes
(port-scan bursts, data exfiltration, DoS floods). Labels are provided ONLY
to evaluate the unsupervised detector -- the model never sees them at fit time.

Synthetic by design: real labeled flow captures (CICIDS, etc.) are large and
licensed. Swap in a real CSV to run the same pipeline on genuine traffic.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

FEATURES = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "pkts_per_sec"]


def generate(n: int = 4000, contamination: float = 0.05, seed: int = 42):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    if (isinstance(contamination, bool) or not isinstance(contamination, Real)
            or not math.isfinite(contamination) or not 0 <= contamination <= 0.5):
        raise ValueError("contamination must be finite and between 0 and 0.5")
    rng = np.random.default_rng(seed)
    n_anom = int(n * contamination)
    n_norm = n - n_anom

    # Normal traffic: short-to-medium flows, balanced-ish bytes.
    dur = rng.lognormal(mean=1.0, sigma=1.0, size=n_norm)
    src_b = rng.lognormal(mean=6.5, sigma=1.0, size=n_norm)
    dst_b = rng.lognormal(mean=6.8, sigma=1.0, size=n_norm)
    src_p = rng.lognormal(mean=2.0, sigma=0.8, size=n_norm)
    dst_p = rng.lognormal(mean=2.2, sigma=0.8, size=n_norm)
    normal = np.column_stack([dur, src_b, dst_b, src_p, dst_p, src_p / np.maximum(dur, 0.01)])

    anoms = []
    for i in range(n_anom):
        kind = i % 3
        if kind == 0:  # port scan: tiny flows, many packets, high pps
            row = [rng.uniform(0.01, 0.1), rng.uniform(40, 120), rng.uniform(0, 40),
                   rng.uniform(30, 100), 0]
        elif kind == 1:  # exfiltration: huge src_bytes, long duration
            row = [rng.uniform(50, 300), rng.lognormal(13, 0.5), rng.uniform(100, 500),
                   rng.uniform(500, 2000), rng.uniform(10, 50)]
        else:  # DoS flood: very high packets both ways, high pps
            row = [rng.uniform(1, 5), rng.lognormal(9, 0.5), rng.uniform(100, 400),
                   rng.uniform(5000, 20000), rng.uniform(10, 40)]
        # Rate is derived from the same packets and duration as normal flows.
        # Independent sampling would make impossible telemetry a shortcut for ML.
        row.append(row[3] / max(row[0], 0.01))
        anoms.append(row)
    anoms = np.array(anoms) if anoms else np.empty((0, len(FEATURES)))

    X = np.vstack([normal, anoms])
    y = np.concatenate([np.zeros(n_norm), np.ones(n_anom)])  # 1 = anomaly
    perm = rng.permutation(len(X))
    return X[perm], y[perm].astype(int)
