# Detection Engineering Notes

The model in this repository is intentionally small, but putting an anomaly detector into a SOC requires more than obtaining a good offline score. This note describes how I would turn the prototype into an operational detection and what I would measure before trusting it.

## What the model contributes

Isolation Forest gives a ranking signal for flows that look unusual relative to the fitted baseline. That is useful for discovering behavior that does not match an existing signature, but "unusual" is not equivalent to "malicious."

The operational output should therefore be treated as a **triage signal**, not an autonomous incident verdict.

## Baseline design

A production baseline should not blindly mix every host and time period. Normal behavior differs by asset role, environment, and time window. Useful segmentation candidates include:

- workstation vs server vs infrastructure assets;
- internal-to-internal vs inbound vs outbound flows;
- business-hours vs overnight behavior;
- high-volume services that would otherwise dominate byte/packet distributions.

The baseline also needs a retraining policy. Updating too slowly causes normal infrastructure changes to generate noise; updating too quickly risks learning persistent attacker behavior as normal.

## Alert enrichment

A raw anomaly score is rarely enough for an analyst. Before creating an alert, I would enrich the flow with available context such as:

- source/destination asset identity and criticality;
- internal/external classification;
- destination reputation or known service ownership;
- authentication or endpoint telemetry near the same timestamp;
- whether the behavior is rare for the individual host as well as globally.

That context lets the anomaly become one feature in an investigation rather than an unexplained number.

## Evaluation beyond ROC-AUC

ROC-AUC is useful during model development but does not answer the SOC question: "How many alerts will analysts actually receive?"

Before deployment I would measure:

1. alerts per 1,000 or 10,000 flows at candidate thresholds;
2. precision in the top-N alerts an analyst could realistically review;
3. false-positive concentration by host/service;
4. stability of scores across time windows;
5. detection coverage for specific behaviors such as scanning, exfiltration, and flooding;
6. time-to-detect relative to a rule-based baseline.

A threshold should be selected against an operational alert budget, not just the point that maximizes F1 on a convenient dataset.

## Drift and feedback

Production traffic changes. I would track feature distributions and anomaly-score distributions over time and alert on material shifts. Analyst dispositions can be retained as evaluation labels, but they should not automatically retrain the model without review because analyst feedback can be sparse and biased toward alerts that were already surfaced.

## Current limitations

The repository currently uses synthetic data with deliberately separable anomaly distributions. That makes the pipeline reproducible but does not establish real-world SOC performance. A stronger next iteration would ingest a public flow dataset, preserve a time-based train/test split, compare against simple statistical baselines, and report alert-volume metrics alongside conventional ML metrics.

The engineering goal is not to claim that unsupervised ML "detects unknown attacks." It is to build a measurable prioritization signal and understand when that signal adds information beyond simpler detection logic.