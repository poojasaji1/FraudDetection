import os
import sys
import json
import logging
import psycopg2
import numpy as np
import pandas as pd

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

MONITORED_FEATURES = [
    "tx_amount_last_1h",
    "tx_count_last_10m",
    "avg_amount_30d",
    "merchant_risk_score",
    "distance_from_home",
    "hour_of_day",
    "is_weekend",
]

_DB = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB", "predictions_db"),
    "user":     os.getenv("POSTGRES_USER", "fraud"),
    "password": os.getenv("POSTGRES_PASSWORD", "changeme"),
}


def load_reference() -> pd.DataFrame:
    path = os.path.join(_PROJECT_DIR, "data", "processed", "features.parquet")
    df = pd.read_parquet(path, columns=MONITORED_FEATURES)
    return df.dropna()


def load_current(hours: int = 24) -> pd.DataFrame:
    sql = f"""
        SELECT all_features
        FROM predictions
        WHERE created_at > NOW() - INTERVAL '{hours} hours'
          AND all_features IS NOT NULL
    """
    try:
        conn = psycopg2.connect(**_DB)
        df_raw = pd.read_sql(sql, conn)
        conn.close()
    except Exception as exc:
        logging.error("Postgres unavailable: %s", exc)
        return pd.DataFrame(columns=MONITORED_FEATURES)

    if df_raw.empty:
        return pd.DataFrame(columns=MONITORED_FEATURES)

    records = []
    for row in df_raw["all_features"]:
        d = json.loads(row) if isinstance(row, str) else (row or {})
        records.append({f: d.get(f) for f in MONITORED_FEATURES})

    return pd.DataFrame(records).dropna()


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = reference.dropna().values
    cur = current.dropna().values
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    breakpoints = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    eps = 1e-4
    ref_pct = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_pct = np.histogram(cur, bins=breakpoints)[0].astype(float)
    ref_pct = np.where(ref_pct == 0, eps, ref_pct) / len(ref)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct) / len(cur)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    if lo == hi:
        return 0.0
    eps = 1e-4
    p_h = np.histogram(p, bins=bins, range=(lo, hi))[0].astype(float)
    q_h = np.histogram(q, bins=bins, range=(lo, hi))[0].astype(float)
    p_h = np.where(p_h == 0, eps, p_h); p_h /= p_h.sum()
    q_h = np.where(q_h == 0, eps, q_h); q_h /= q_h.sum()
    return float(np.sum(p_h * np.log(p_h / q_h)))


def _current_fraud_scores(hours: int = 24) -> np.ndarray:
    sql = f"""
        SELECT fraud_score FROM predictions
        WHERE created_at > NOW() - INTERVAL '{hours} hours'
          AND fraud_score IS NOT NULL
    """
    try:
        conn = psycopg2.connect(**_DB)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df["fraud_score"].dropna().values
    except Exception:
        return np.array([])


def _log_event(drift_detected: bool, feature_psi: dict, score_kl: float, triggered: list):
    sql = """
        INSERT INTO drift_events (drift_detected, feature_psi, score_kl, triggered_features)
        VALUES (%s, %s, %s, %s)
    """
    try:
        conn = psycopg2.connect(**_DB)
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    drift_detected,
                    json.dumps(feature_psi),
                    score_kl,
                    json.dumps(triggered),
                ))
        conn.close()
    except Exception as exc:
        logging.error("Failed to log drift event: %s", exc)


def run_drift_check() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    reference = load_reference()
    current   = load_current(hours=24)

    if len(current) < 100:
        result = {
            "drift_detected": False,
            "reason": "insufficient_data",
            "current_rows": len(current),
        }
        print(f"Drift check skipped — only {len(current)} rows in last 24h (need ≥100)")
        return result

    feature_psi: dict[str, float] = {}
    print(f"\n{'Feature':<30} {'PSI':>8}  Status")
    print("─" * 50)
    for feat in MONITORED_FEATURES:
        psi = compute_psi(reference[feat], current[feat])
        feature_psi[feat] = round(psi, 4)
        status = "DRIFT" if psi > 0.2 else ("WARN" if psi > 0.1 else "OK")
        print(f"{feat:<30} {psi:>8.4f}  {status}")

    cur_scores = _current_fraud_scores(hours=24)
    if len(cur_scores) >= 2:
        ref_uniform = np.random.uniform(0, 1, len(cur_scores))
        score_kl = _kl_divergence(cur_scores, ref_uniform)
    else:
        score_kl = 0.0

    triggered     = [f for f, psi in feature_psi.items() if psi > 0.2]
    drift_detected = len(triggered) > 0

    print(f"\n{'Score KL divergence':<30} {score_kl:>8.4f}")
    print(f"{'Drift detected':<30} {'YES ⚠' if drift_detected else 'NO ✓':>8}")
    if triggered:
        print(f"Triggered features: {triggered}")

    _log_event(drift_detected, feature_psi, score_kl, triggered)

    return {
        "drift_detected":       drift_detected,
        "feature_psi":          feature_psi,
        "score_kl_divergence":  round(score_kl, 4),
        "triggered_features":   triggered,
    }


if __name__ == "__main__":
    import json as _json
    result = run_drift_check()
    print("\nResult:", _json.dumps(result, indent=2))
