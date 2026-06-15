"""
Insert 500 synthetic prediction rows with a deliberately shifted tx_amount_last_1h
distribution so that drift_detector.py can detect PSI > 0.2.
"""
import os
import sys
import json
import random
import logging
import psycopg2
import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

_DB = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB", "predictions_db"),
    "user":     os.getenv("POSTGRES_USER", "fraud"),
    "password": os.getenv("POSTGRES_PASSWORD", "changeme"),
}

N = 500
rng = np.random.default_rng(42)


def _build_features(i: int) -> dict:
    # tx_amount_last_1h shifted 5× higher to trigger PSI > 0.2
    return {
        "tx_amount_last_1h":    float(rng.exponential(scale=2500)),   # ref: ~500
        "tx_count_last_10m":    int(rng.integers(1, 30)),
        "avg_amount_30d":       float(rng.uniform(100, 4000)),
        "merchant_risk_score":  float(rng.uniform(0, 1)),
        "distance_from_home":   float(rng.exponential(scale=150)),
        "hour_of_day":          int(rng.integers(0, 24)),
        "is_weekend":           int(rng.integers(0, 2)),
    }


def insert_drifted_rows():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sql = """
        INSERT INTO predictions
            (user_id, amount, fraud_score, label, latency_ms, model_version, all_features)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
    """

    try:
        conn = psycopg2.connect(**_DB)
    except Exception as exc:
        print(f"Cannot connect to Postgres: {exc}")
        sys.exit(1)

    rows = []
    for i in range(N):
        features = _build_features(i)
        amount   = features["tx_amount_last_1h"]
        score    = float(rng.uniform(0.5, 1.0))     # push fraud scores high
        label    = "fraud" if score >= 0.5 else "legit"
        rows.append((
            f"sim_user_{i:04d}",
            round(amount, 2),
            round(score, 4),
            label,
            round(float(rng.uniform(10, 80)), 2),
            "sim",
            json.dumps(features),
        ))

    with conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    conn.close()

    print(f"Inserted {N} drifted rows into predictions table")
    print("tx_amount_last_1h shifted 5× — run `make drift` to confirm PSI > 0.2")


if __name__ == "__main__":
    insert_drifted_rows()
