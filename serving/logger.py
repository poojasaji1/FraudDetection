import os
import sys
import json
import threading
import psycopg2

_DB_PARAMS = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB", "predictions_db"),
    "user":     os.getenv("POSTGRES_USER", "fraud"),
    "password": os.getenv("POSTGRES_PASSWORD", "changeme"),
}

_INSERT_SQL = """
    INSERT INTO predictions
        (user_id, amount, fraud_score, label, latency_ms, model_version, all_features)
    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
"""


def _insert(user_id, amount, fraud_score, label, latency_ms, model_version, features):
    try:
        conn = psycopg2.connect(**_DB_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute(_INSERT_SQL, (
                    user_id, amount, fraud_score, label,
                    latency_ms, model_version, json.dumps(features),
                ))
        conn.close()
    except Exception as exc:
        print(f"[logger] prediction log failed: {exc}", file=sys.stderr)


def log_prediction(
    user_id: str,
    amount: float,
    fraud_score: float,
    label: str,
    latency_ms: float,
    model_version: str,
    features: dict,
) -> None:
    t = threading.Thread(
        target=_insert,
        args=(user_id, amount, fraud_score, label, latency_ms, model_version, features),
        daemon=True,
    )
    t.start()
