1# Fraud Detection MLOps Platform

End-to-end MLOps pipeline for real-time fraud detection using XGBoost, Feast, MLflow, Ray Serve, Airflow, and Evidently.
11111
## Quick start

```bash
cp .env.example .env
make setup      # install Python dependencies
make services   # start all Docker services
make data       # validate and split raw data
```

## Services

| Service    | URL                    |
|------------|------------------------|
| MLflow     | http://localhost:5000  |
| Airflow    | http://localhost:8080  |
| Grafana    | http://localhost:3000  |
| Prometheus | http://localhost:9090  |
| Redis      | localhost:6379         |
| Postgres   | localhost:5432         |

## Data

Place (or symlink) `train_transaction.csv` and `train_identity.csv` in `data/raw/` before running `make data`.

## Project layout

```
data/        — raw inputs, processed parquets, validation script
features/    — Feast feature store definitions
training/    — XGBoost training pipeline and Airflow DAGs
registry/    — MLflow model registry helpers
serving/     — Ray Serve inference API
monitoring/  — Evidently drift detection and Prometheus metrics
tests/       — integration and load tests
```
