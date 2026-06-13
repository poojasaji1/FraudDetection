import os
import sys
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TRAINING_DIR)

MODEL_NAME = "fraud_detector"

_C_COLS  = [f"C{i}"  for i in range(1, 15)]
_V_COLS  = [f"V{i}"  for i in range(1, 340)]
_D_COLS  = [f"D{i}"  for i in range(1, 16)]
_ID_COLS = ["id_01","id_02","id_03","id_04","id_05","id_06","id_07","id_08","id_09",
            "id_10","id_11","id_13","id_14","id_17","id_18","id_19","id_20","id_21",
            "id_22","id_24","id_25","id_26","id_32"]
FEATURE_COLS = [
    "tx_amount_last_1h", "tx_count_last_10m", "avg_amount_30d",
    "merchant_risk_score", "distance_from_home", "hour_of_day",
    "is_weekend", "TransactionAmt",
] + _C_COLS + _V_COLS + _D_COLS + _ID_COLS


def register(run_id: str, metrics: dict) -> bool:
    """Gate-check against Production AUC-PR, then register to Staging if gate passes."""
    client = MlflowClient()
    new_auc_pr = metrics.get("auc_pr", 0.0)

    # Check current Production baseline
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        prod_versions = []

    if prod_versions:
        prod_run = client.get_run(prod_versions[0].run_id)
        prod_auc_pr = prod_run.data.metrics.get("auc_pr", 0.0)
        if new_auc_pr <= prod_auc_pr:
            client.set_tag(run_id, "promoted", "false")
            print(
                f"Gate failed: new AUC-PR {new_auc_pr:.4f} "
                f"<= production {prod_auc_pr:.4f} — not registering"
            )
            return False

    # Register and move to Staging
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.transition_model_version_stage(MODEL_NAME, mv.version, "Staging")

    client.set_model_version_tag(
        MODEL_NAME, mv.version, "training_date",
        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    )
    client.set_model_version_tag(MODEL_NAME, mv.version, "auc_pr", f"{new_auc_pr:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "feature_list", ",".join(FEATURE_COLS))
    client.set_tag(run_id, "promoted", "true")

    print(f"Registered version {mv.version} → Staging  (AUC-PR {new_auc_pr:.4f})")
    return True
