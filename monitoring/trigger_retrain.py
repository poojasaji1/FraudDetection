"""
Trigger an Airflow DAG run when drift is detected.
Called by drift_detector.run_drift_check() or as a standalone script.
"""
import os
import sys
import json
import logging
import requests

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

AIRFLOW_HOST = os.getenv("AIRFLOW_HOST", "http://localhost:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")
DAG_ID       = "retrain_fraud_model"


def trigger_retraining(reason: str = "drift_detected", metadata: dict | None = None) -> bool:
    url  = f"{AIRFLOW_HOST}/api/v1/dags/{DAG_ID}/dagRuns"
    body = {
        "conf": {
            "reason":   reason,
            "metadata": metadata or {},
        }
    }
    try:
        resp = requests.post(
            url,
            json=body,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=10,
        )
        if resp.status_code in (200, 201):
            run_id = resp.json().get("dag_run_id", "unknown")
            logging.info("Triggered DAG %s — run_id: %s", DAG_ID, run_id)
            print(f"Retraining triggered — dag_run_id: {run_id}")
            return True
        else:
            logging.error("Airflow returned %d: %s", resp.status_code, resp.text)
            print(f"Failed to trigger retraining: HTTP {resp.status_code}")
            return False
    except Exception as exc:
        logging.error("Cannot reach Airflow: %s", exc)
        print(f"Cannot reach Airflow at {AIRFLOW_HOST}: {exc}")
        return False


def run(drift_result: dict) -> bool:
    if not drift_result.get("drift_detected"):
        print("No drift detected — retraining not triggered")
        return False

    triggered = drift_result.get("triggered_features", [])
    score_kl  = drift_result.get("score_kl_divergence", 0.0)
    print(f"Drift confirmed on {triggered} (KL={score_kl:.4f}) — triggering retraining …")
    return trigger_retraining(
        reason="drift_detected",
        metadata={
            "triggered_features": triggered,
            "score_kl":           score_kl,
            "feature_psi":        drift_result.get("feature_psi", {}),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from monitoring.drift_detector import run_drift_check
    result = run_drift_check()
    run(result)
