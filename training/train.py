import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TRAINING_DIR)
_FEATURES_DIR = os.path.join(_PROJECT_DIR, "features")

for _p in (_TRAINING_DIR, _FEATURES_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from offline_store import get_training_features
from evaluate import evaluate
from register import register

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

TRAIN_PARQUET = os.path.join(_PROJECT_DIR, "data", "processed", "train.parquet")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Engineered features (from Feast feature store)
_FEAST_FEATURES = [
    "tx_amount_last_1h", "tx_count_last_10m", "avg_amount_30d",
    "merchant_risk_score", "distance_from_home", "hour_of_day",
    "is_weekend", "TransactionAmt",
]

# Raw transaction features; XGBoost handles NaN natively — no fillna needed for V/D/id columns
_C_COLS  = [f"C{i}"  for i in range(1, 15)]    # C1-C14
_V_COLS  = [f"V{i}"  for i in range(1, 340)]   # V1-V339
_D_COLS  = [f"D{i}"  for i in range(1, 16)]    # D1-D15 time-delta features
_ID_COLS = [
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
    "id_07", "id_08", "id_09", "id_10", "id_11", "id_13",
    "id_14", "id_17", "id_18", "id_19", "id_20", "id_21",
    "id_22", "id_24", "id_25", "id_26", "id_32",
]
_EXTRA_COLS = [
    "card2", "card3", "card5", "dist1",
    "addr1", "addr2",                          # raw billing address codes
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "ProductCD_enc", "card4_enc", "card6_enc",
    "P_email_risk", "R_email_risk",            # email domain fraud rates
    "uid_risk",                                # fraud rate per (card1, addr1) user proxy
    "addr1_risk",                              # fraud rate per addr1 billing location
    "card1_risk",                              # fraud rate per card1
    "card1_freq",                              # transaction count per card1
    "card1_mean_amt",                          # mean TransactionAmt per card1
    "amt_dev_from_card1",                      # TransactionAmt - card1 mean (anomaly signal)
    "uid2_risk",                               # fraud rate per (card1, card2) pair
]

FEATURE_COLS = _FEAST_FEATURES + _C_COLS + _V_COLS + _D_COLS + _ID_COLS + _EXTRA_COLS

# M column encodings
_M_TF_MAP = {"T": 1.0, "F": 0.0}
_M4_MAP   = {"M0": 0.0, "M1": 1.0, "M2": 2.0}

# Ordinal encodings for low-cardinality categoricals
_PRODUCT_MAP = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
_CARD4_MAP   = {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}
_CARD6_MAP   = {"debit": 0, "credit": 1, "debit or credit": 2, "charge card": 3}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 8,
    "learning_rate": 0.01,
    "n_estimators": 4000,
    "early_stopping_rounds": 150,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
}


def train(run_name: str = None) -> str:
    """Full training pipeline: load → enrich → split → fit → log → register.

    Returns the MLflow run_id for downstream tasks.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("fraud_detection")

    # 1-2. Load and enrich
    print("Loading training data ...")
    df = pd.read_parquet(TRAIN_PARQUET)
    df = get_training_features(df)

    # 3. Drop rows where engineered features are null; encode / fill raw columns
    df = df.dropna(subset=_FEAST_FEATURES)
    df[_C_COLS] = df[_C_COLS].fillna(0)
    for col in ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]:
        df[col] = df[col].map(_M_TF_MAP)
    df["M4"] = df["M4"].map(_M4_MAP)

    # Ordinal encode low-cardinality categoricals
    df["ProductCD_enc"] = df["ProductCD"].map(_PRODUCT_MAP)
    df["card4_enc"]     = df["card4"].map(_CARD4_MAP)
    df["card6_enc"]     = df["card6"].map(_CARD6_MAP)

    # Target-encode email domains (mean isFraud per domain, computed on full training set)
    global_fraud_rate = df["isFraud"].mean()
    p_risk = df.groupby("P_emaildomain")["isFraud"].mean()
    r_risk = df.groupby("R_emaildomain")["isFraud"].mean()
    df["P_email_risk"] = df["P_emaildomain"].map(p_risk).fillna(global_fraud_rate)
    df["R_email_risk"] = df["R_emaildomain"].map(r_risk).fillna(global_fraud_rate)

    # uid = card1 + addr1: proxy for a unique user; target-encode fraud rate
    df["_uid"] = df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(int).astype(str)
    uid_risk   = df.groupby("_uid")["isFraud"].mean()
    df["uid_risk"] = df["_uid"].map(uid_risk).fillna(global_fraud_rate)

    # addr1 billing location risk
    addr1_risk = df.groupby("addr1")["isFraud"].mean()
    df["addr1_risk"] = df["addr1"].map(addr1_risk).fillna(global_fraud_rate)

    # card1-level encodings
    card1_freq = df.groupby("card1")["TransactionID"].transform("count")
    df["card1_freq"] = card1_freq
    card1_risk = df.groupby("card1")["isFraud"].mean()
    df["card1_risk"] = df["card1"].map(card1_risk).fillna(global_fraud_rate)
    card1_mean_amt = df.groupby("card1")["TransactionAmt"].mean()
    df["card1_mean_amt"] = df["card1"].map(card1_mean_amt).fillna(df["TransactionAmt"].mean())
    df["amt_dev_from_card1"] = df["TransactionAmt"] - df["card1_mean_amt"]

    # uid2 = card1 + card2: finer-grained user proxy
    df["_uid2"] = df["card1"].astype(str) + "_" + df["card2"].fillna(-1).astype(int).astype(str)
    uid2_risk = df.groupby("_uid2")["isFraud"].mean()
    df["uid2_risk"] = df["_uid2"].map(uid2_risk).fillna(global_fraud_rate)

    print(f"Training rows after preprocessing: {len(df):,}")

    # 4. Split X / y
    X = df[FEATURE_COLS]
    y = df["isFraud"]

    # 5. Stratified train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Fraud rate train={y_train.mean():.4f} val={y_val.mean():.4f}")

    # 6. Class imbalance weight
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 9. Log params
        mlflow.log_params(params)

        # 7. Train
        print("Training XGBoost ...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # 10. Log feature importances
        importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        mlflow.log_dict(importances, "feature_importances.json")

        # 11. Log model artifact
        mlflow.xgboost.log_model(model, "model")

        # 12. Evaluate and log metrics
        metrics = evaluate(model, X_val, y_val, run_id)

        # 13. Register (gate + staging)
        register(run_id, metrics)

        return run_id


if __name__ == "__main__":
    run_id = train(run_name="baseline")
    print(f"\nDone. MLflow run_id: {run_id}")
