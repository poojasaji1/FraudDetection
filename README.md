# Fraud Detection MLOps Platform

A production-grade fraud detection system built in 7 days. It classifies payment transactions in real time using an XGBoost model trained on 472,000 labelled transactions, served behind a FastAPI endpoint that pulls live card features from a Redis-backed Feast online store. What makes this hard is not the model — it is everything else: feature pipelines that never leak future data into training, class imbalance so severe that 96% negative means a useless classifier, a rollback mechanism that swaps models without downtime, drift detection that triggers automated retraining before the model degrades in production, and all of it deployed to Kubernetes with autoscaling and alerting.

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   OFFLINE PATH          │              ONLINE PATH                      │
                         │                                              │
  Raw CSVs               │  POST /v1/predict                            │
     │                   │         │                                    │
     ▼                   │         ▼                                    │
  Feature                │    FastAPI server (uvicorn)                  │
  Engineering ──────────►│         │                                    │
  (PIT correct)          │         ├──► Feast online store (Redis)     │
     │                   │         │         └── 7 card features        │
     ▼                   │         │                                    │
  features.parquet       │         ├──► XGBoost model (MLflow)         │
     │                   │         │         └── fraud_score [0,1]      │
     ▼                   │         │                                    │
  Feast materialize      │         └──► Postgres logger (async)        │
  ──────────────────────►│               └── predictions table          │
       Redis             │                                              │
                         └──────────────────────────────────────────────┘

   TRAINING PATH                    MONITORING PATH

   train.parquet                     Postgres predictions
        │                                   │
        ▼                                   ▼
   XGBoost (focal loss)             Drift Detector (hourly)
        │                            PSI per feature + KL div
        ▼                                   │ drift_detected?
   MLflow registry                          ▼
   ┌──────────────┐                  Airflow DAG trigger
   │  Production  │◄── rollback()    retrain_fraud_model
   │  Archived    │
   └──────────────┘
        │
        ▼
   Kubernetes HPA
   (2–5 replicas, CPU 70%)
```

## Stack

| Layer | Tool | Why |
|---|---|---|
| Model | XGBoost + focal loss | Handles class imbalance (3.5% fraud rate) without oversampling |
| Feature store | Feast + Redis | Point-in-time correctness offline; sub-5ms online retrieval |
| Model registry | MLflow | Versioned artifacts, Production/Archived stages, one-line rollback |
| Serving | FastAPI + uvicorn | Lightweight; Ray Serve OOMs at 512Mi; same API surface |
| Orchestration | Kubernetes + HPA | Auto-scales 2→5 replicas on CPU pressure |
| Drift detection | PSI + KL divergence | PSI > 0.2 on any feature triggers retraining |
| Retraining | Airflow DAG | Scheduled + on-demand via REST API |
| Observability | Prometheus + Grafana | 4 alert rules; 6-panel dashboard |
| Database | Postgres | Prediction logging + drift event history |

## Quickstart

```bash
git clone https://github.com/poojasaji1/FraudDetection.git
cd FraudDetection

cp .env.example .env          # set credentials if needed

make setup                    # install Python dependencies
make services                 # start Postgres, Redis, MLflow, Airflow (Docker)
make data                     # validate raw CSVs
make features                 # build Feast feature views + materialize to Redis
make train                    # train XGBoost, log to MLflow, promote to Production
make serve                    # start FastAPI server on localhost:8000

# Smoke test
curl -s -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123456",
    "amount": 1500.00,
    "merchant_id": "M_ELECTRONICS",
    "timestamp": "2024-01-15T14:30:00",
    "location": {"lat": 37.7, "lng": -122.4}
  }' | python -m json.tool
```

Expected response:
```json
{
  "fraud_score": 0.9943,
  "label": "fraud",
  "latency_ms": 56.2,
  "model_version": "8"
}
```

### Drift detection demo

```bash
make simulate-drift    # insert 500 drifted rows (tx_amount shifted 5x)
make drift             # run PSI check — prints per-feature PSI, triggers Airflow
```

### Rollback demo

```bash
make rollback          # swap Production <-> Archived in MLflow registry
curl http://localhost:8000/health   # restart server to see new model_version
```

### Kubernetes deploy

```bash
make k8s-deploy        # build image, apply manifests, wait for rollout
make k8s-status        # check pods, HPA, service
make k8s-monitoring    # apply drift CronJob
make load-test         # run Locust at 1000 users — watch HPA scale up
```

## Benchmarks

All numbers measured on a 2023 MacBook Pro (M2, 16 GB RAM) with services running in Docker.

| Metric | Measured | Target |
|---|---|---|
| p99 prediction latency | 56 ms | < 100 ms |
| Online feature retrieval | < 5 ms | < 5 ms |
| Model AUC-PR | 0.877 | > 0.85 |
| Rollback time | < 1 s | < 30 s |
| Drift detection to Airflow trigger | < 65 s | < 1 hr |

## Hard Problems

### Point-in-time correctness

When training, it is easy to accidentally compute features using data that would not have been available at transaction time — for example, aggregating all of a card's historical amounts including transactions that happened *after* the one being labelled. This leaks the future into training and inflates offline metrics while the model fails in production. The feature pipeline here computes rolling aggregates (tx_amount_last_1h, tx_count_last_10m, avg_amount_30d) using only events strictly before each transaction's timestamp, joining onto a sorted event log. Feast's FileSource with `timestamp_field` enforces this contract; any ad-hoc merge on card1 without a time cutoff would silently break it.

### Class imbalance

Only 3.5% of transactions are fraudulent. A model that always predicts "legitimate" achieves 96.5% accuracy and is completely useless. This system uses focal loss (`scale_pos_weight` weighted by class ratio) to force the model to pay attention to the minority class, and evaluates on AUC-PR rather than accuracy or AUC-ROC. AUC-PR penalises the model heavily for missed frauds at high precision thresholds — the metric that actually reflects business cost.

### Delayed ground truth

Chargebacks (confirmed fraud labels) typically arrive 30–90 days after the transaction. At training time, recent transactions have unconfirmed labels, which means the training window must lag reality or risk contaminating the label set. This system sidesteps the problem by using a static labelled dataset, but in production the retraining DAG would need a 90-day label delay baked in — otherwise freshly trained models are trained on optimistic labels and degrade once ground truth arrives.

### Adversarial drift

Not all distribution shift is passive. Fraudsters actively probe model thresholds — if they detect that transactions under $500 are rarely flagged, they fragment large fraudulent purchases. PSI-based drift detection catches this when the tx_amount distribution shifts, but it cannot distinguish legitimate seasonality from adversarial probing. A production system would layer rules (velocity checks, IP reputation, device fingerprinting) that are harder to probe than a single ML score.

### Rollback safety

Naively, "rollback" means "deploy the previous model." But if the previous model was archived because it degraded, re-promoting it will reproduce the same problem. This system's rollback is safe only because the Archived model was promoted-then-demoted manually, not because it failed silently. In production, rollback must be gated on a shadow evaluation — the candidate model should be shadow-scored against live traffic for at least 30 minutes before promotion, and the rollback target must have its own held-out evaluation logged before it is archived.

## What I'd do at true production scale

- **Real-time feature computation**: Replace batch Feast materialisation with a Kafka consumer that updates Redis on every transaction event — eliminates the staleness window where rolling aggregates are hours old.
- **Label pipeline**: Build an automated chargeback ingestion job that joins dispute records back to prediction IDs after a 90-day window and writes confirmed labels to a training table, so the retraining DAG always has clean ground truth.
- **Shadow scoring on every deploy**: Route 5% of production traffic to the new model version in parallel with the current Production model, compare fraud-rate distributions, and only promote if KS-test p-value > 0.05.
- **Multi-model ensemble**: XGBoost handles tabular features well but misses sequential patterns (burst fraud across multiple cards in seconds). Add a lightweight GRU or sliding-window model on card-level event sequences, ensemble the two scores.
- **Threshold calibration per merchant**: A 0.5 fraud threshold loses money for high-value merchants (where false negatives cost more) and over-blocks for low-value ones (where false positives lose customers). Calibrate per merchant segment using Platt scaling on held-out validation data.
