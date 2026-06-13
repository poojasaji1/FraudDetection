import os
import sys
import time
import math

import numpy as np
import pandas as pd

_SERVING_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_SERVING_DIR)
_FEATURES_DIR = os.path.join(_PROJECT_DIR, "features")

for _p in (_SERVING_DIR, _FEATURES_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online_store import get_online_features
from logger import log_prediction

# Must exactly match training/train.py FEATURE_COLS
_FEAST_FEATURES = [
    "tx_amount_last_1h", "tx_count_last_10m", "avg_amount_30d",
    "merchant_risk_score", "distance_from_home", "hour_of_day",
    "is_weekend", "TransactionAmt",
]
_C_COLS  = [f"C{i}"  for i in range(1, 15)]
_V_COLS  = [f"V{i}"  for i in range(1, 340)]
_D_COLS  = [f"D{i}"  for i in range(1, 16)]
_ID_COLS = [
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
    "id_07", "id_08", "id_09", "id_10", "id_11", "id_13",
    "id_14", "id_17", "id_18", "id_19", "id_20", "id_21",
    "id_22", "id_24", "id_25", "id_26", "id_32",
]
_EXTRA_COLS = [
    "card2", "card3", "card5", "dist1",
    "addr1", "addr2",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "ProductCD_enc", "card4_enc", "card6_enc",
    "P_email_risk", "R_email_risk",
    "uid_risk", "addr1_risk",
    "card1_risk", "card1_freq", "card1_mean_amt", "amt_dev_from_card1",
    "uid2_risk",
]
FEATURE_COLS = _FEAST_FEATURES + _C_COLS + _V_COLS + _D_COLS + _ID_COLS + _EXTRA_COLS


def _nan_safe(val) -> object:
    """Convert float NaN to None for JSON serialisation."""
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def predict(
    user_id: str,
    amount: float,
    merchant_id: str,
    timestamp: str,
    location: dict,
    model,
    model_version: str,
) -> dict:
    start = time.time()

    try:
        online_feats = get_online_features(card1=int(user_id))
    except Exception:
        online_feats = {}

    # Seed every column with NaN; XGBoost handles missing natively
    row = {col: np.nan for col in FEATURE_COLS}
    for k, v in online_feats.items():
        if k in row:
            row[k] = v
    row["TransactionAmt"] = amount

    X = pd.DataFrame([row])[FEATURE_COLS]

    proba = model.predict_proba(X)[0]
    fraud_score = float(proba[1])
    label = "fraud" if fraud_score >= 0.5 else "legitimate"
    latency_ms = (time.time() - start) * 1000

    log_prediction(
        user_id=user_id,
        amount=amount,
        fraud_score=fraud_score,
        label=label,
        latency_ms=round(latency_ms, 2),
        model_version=model_version,
        features={k: _nan_safe(v) for k, v in row.items()},
    )

    return {
        "fraud_score": round(fraud_score, 4),
        "label": label,
        "latency_ms": round(latency_ms, 2),
        "model_version": model_version,
    }
