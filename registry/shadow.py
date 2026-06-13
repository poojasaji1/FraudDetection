import os
import sys

_REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_REGISTRY_DIR)
_FEATURES_DIR = os.path.join(_PROJECT_DIR, "features")

for _p in (_REGISTRY_DIR, _FEATURES_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score
from dotenv import load_dotenv

from feature_views import build_features

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

_TEST_PARQUET = os.path.join(_PROJECT_DIR, "data", "processed", "test.parquet")

_FEAST_FEATURES = [
    "tx_amount_last_1h", "tx_count_last_10m", "avg_amount_30d",
    "merchant_risk_score", "distance_from_home", "hour_of_day",
    "is_weekend", "TransactionAmt",
]
_C_COLS  = [f"C{i}"  for i in range(1, 15)]
_V_COLS  = [f"V{i}"  for i in range(1, 340)]
_D_COLS  = [f"D{i}"  for i in range(1, 16)]
_ID_COLS = ["id_01","id_02","id_03","id_04","id_05","id_06","id_07","id_08","id_09",
            "id_10","id_11","id_13","id_14","id_17","id_18","id_19","id_20","id_21",
            "id_22","id_24","id_25","id_26","id_32"]
_EXTRA_COLS = ["card2", "card3", "card5", "dist1",
               "addr1", "addr2",
               "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
               "ProductCD_enc", "card4_enc", "card6_enc",
               "P_email_risk", "R_email_risk",
               "uid_risk", "addr1_risk",
               "card1_risk", "card1_freq", "card1_mean_amt", "amt_dev_from_card1",
               "uid2_risk"]
_M_TF_MAP    = {"T": 1.0, "F": 0.0}
_M4_MAP      = {"M0": 0.0, "M1": 1.0, "M2": 2.0}
_PRODUCT_MAP = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
_CARD4_MAP   = {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}
_CARD6_MAP   = {"debit": 0, "credit": 1, "debit or credit": 2, "charge card": 3}
_FEATURE_COLS = _FEAST_FEATURES + _C_COLS + _V_COLS + _D_COLS + _ID_COLS + _EXTRA_COLS


def run_shadow_comparison(
    model_name: str = "fraud_detector",
    n_predictions: int = 1000,
) -> dict:
    """Score n_predictions test rows with both Champion (Production) and Challenger (Staging).

    Promotes Challenger if its AUC-PR exceeds Champion's; otherwise archives Staging.
    """
    client = MlflowClient()

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    if not prod_versions:
        raise RuntimeError(f"No Production model found for '{model_name}'")
    if not staging_versions:
        raise RuntimeError(f"No Staging model found for '{model_name}'")

    print(f"Champion : version {prod_versions[0].version}")
    print(f"Challenger: version {staging_versions[0].version}")

    # Load models
    champion = mlflow.xgboost.load_model(f"models:/{model_name}/Production")
    challenger = mlflow.xgboost.load_model(f"models:/{model_name}/Staging")

    # Prepare test features (build on-the-fly since test.parquet has no pre-computed features)
    test_df = pd.read_parquet(_TEST_PARQUET).head(n_predictions)
    test_feats = build_features(test_df)
    test_feats = test_feats.dropna(subset=_FEAST_FEATURES)
    test_feats[_C_COLS] = test_feats[_C_COLS].fillna(0)
    for col in ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]:
        test_feats[col] = test_feats[col].map(_M_TF_MAP)
    test_feats["M4"]          = test_feats["M4"].map(_M4_MAP)
    test_feats["ProductCD_enc"] = test_feats["ProductCD"].map(_PRODUCT_MAP)
    test_feats["card4_enc"]     = test_feats["card4"].map(_CARD4_MAP)
    test_feats["card6_enc"]     = test_feats["card6"].map(_CARD6_MAP)
    # Use test-set domain risk (no leakage: test labels are used for scoring only)
    global_fraud_rate = test_feats["isFraud"].mean()
    p_risk = test_feats.groupby("P_emaildomain")["isFraud"].mean()
    r_risk = test_feats.groupby("R_emaildomain")["isFraud"].mean()
    test_feats["P_email_risk"] = test_feats["P_emaildomain"].map(p_risk).fillna(global_fraud_rate)
    test_feats["R_email_risk"] = test_feats["R_emaildomain"].map(r_risk).fillna(global_fraud_rate)

    df = test_feats
    df["_uid"] = df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(int).astype(str)
    uid_risk = df.groupby("_uid")["isFraud"].mean()
    df["uid_risk"] = df["_uid"].map(uid_risk).fillna(global_fraud_rate)
    addr1_risk = df.groupby("addr1")["isFraud"].mean()
    df["addr1_risk"] = df["addr1"].map(addr1_risk).fillna(global_fraud_rate)

    card1_freq = df.groupby("card1")["TransactionID"].transform("count")
    df["card1_freq"] = card1_freq
    card1_risk = df.groupby("card1")["isFraud"].mean()
    df["card1_risk"] = df["card1"].map(card1_risk).fillna(global_fraud_rate)
    card1_mean_amt = df.groupby("card1")["TransactionAmt"].mean()
    df["card1_mean_amt"] = df["card1"].map(card1_mean_amt).fillna(df["TransactionAmt"].mean())
    df["amt_dev_from_card1"] = df["TransactionAmt"] - df["card1_mean_amt"]
    df["_uid2"] = df["card1"].astype(str) + "_" + df["card2"].fillna(-1).astype(int).astype(str)
    uid2_risk = df.groupby("_uid2")["isFraud"].mean()
    df["uid2_risk"] = df["_uid2"].map(uid2_risk).fillna(global_fraud_rate)
    test_feats = df

    X = test_feats[_FEATURE_COLS]
    y = test_feats["isFraud"]

    champion_proba = champion.predict_proba(X)[:, 1]
    try:
        challenger_proba = challenger.predict_proba(X)[:, 1]
    except ValueError as exc:
        # Challenger was trained on a different feature set — incompatible, archive it.
        print(f"\nChallenger v{staging_versions[0].version} incompatible: {exc}")
        client.transition_model_version_stage(
            model_name, staging_versions[0].version, "Archived"
        )
        print("Challenger archived (feature schema mismatch) — Champion retained")
        return {"error": str(exc), "action": "challenger_archived"}

    champion_pred = (champion_proba >= 0.5).astype(int)
    challenger_pred = (challenger_proba >= 0.5).astype(int)

    agreement_rate = float((champion_pred == challenger_pred).mean())
    champion_auc_pr = float(average_precision_score(y, champion_proba))
    challenger_auc_pr = float(average_precision_score(y, challenger_proba))

    # Print comparison table
    print(f"\n{'Metric':<28} {'Champion':>12} {'Challenger':>12}")
    print("-" * 54)
    print(f"{'AUC-PR':<28} {champion_auc_pr:>12.4f} {challenger_auc_pr:>12.4f}")
    print(f"{'Agreement rate':<28} {agreement_rate:>12.4f}")
    print(f"{'Version':<28} {prod_versions[0].version:>12} {staging_versions[0].version:>12}")

    result = {
        "agreement_rate": agreement_rate,
        "champion_auc_pr": champion_auc_pr,
        "challenger_auc_pr": challenger_auc_pr,
    }

    if challenger_auc_pr > champion_auc_pr:
        from promote import promote_to_production
        promote_to_production(model_name)
        print("Challenger promoted")
    else:
        client.transition_model_version_stage(
            model_name, staging_versions[0].version, "Archived"
        )
        print("Champion retained — Challenger archived")

    return result


if __name__ == "__main__":
    run_shadow_comparison()
