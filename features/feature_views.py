import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

# Make this directory importable regardless of invocation path so feast apply works
_FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)

from feast import FeatureView, FileSource, Field
from feast.types import Float64, Int64
from entities import card

_PROJECT_DIR = os.path.dirname(_FEATURES_DIR)
FEATURES_PARQUET = os.path.join(_PROJECT_DIR, "data", "processed", "features.parquet")
TRAIN_PARQUET = os.path.join(_PROJECT_DIR, "data", "processed", "train.parquet")

# ── Feast objects (module-level so `feast apply` registers them) ──────────────

_source = FileSource(
    path=FEATURES_PARQUET,
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[card],
    ttl=timedelta(days=90),
    source=_source,
    schema=[
        Field(name="tx_amount_last_1h", dtype=Float64),
        Field(name="tx_count_last_10m", dtype=Float64),
        Field(name="avg_amount_30d", dtype=Float64),
        Field(name="merchant_risk_score", dtype=Float64),
        Field(name="distance_from_home", dtype=Float64),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
    ],
)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 7 features from a transaction DataFrame.

    Requires columns: TransactionDT, TransactionAmt, card1, addr1, isFraud.
    Returns df sorted by (card1, event_timestamp) with feature columns appended.
    """
    df = df.copy()
    df["event_timestamp"] = (
        pd.to_datetime(df["TransactionDT"], unit="s", origin="2017-11-30")
        .dt.tz_localize("UTC")
    )

    # Sort so groupby+rolling values align with df_sorted rows in order
    df_sorted = df.sort_values(["card1", "event_timestamp", "TransactionID"]).reset_index(drop=True)
    df_temp = df_sorted.set_index("event_timestamp")

    grouped = df_temp.groupby("card1", sort=True)

    # Time-based rolling windows (closed='left' excludes the current row)
    df_sorted["tx_amount_last_1h"] = (
        grouped["TransactionAmt"].rolling("1h", closed="left").sum().values
    )
    df_sorted["tx_count_last_10m"] = (
        grouped["TransactionAmt"].rolling("10min", closed="left").count().values
    )
    df_sorted["avg_amount_30d"] = (
        grouped["TransactionAmt"].rolling("30d", closed="left").mean().values
    )

    # Fill NaN at the start of each card's history (no prior rows in window)
    for col in ("tx_amount_last_1h", "tx_count_last_10m", "avg_amount_30d"):
        df_sorted[col] = df_sorted[col].fillna(0.0)

    # merchant_risk_score — mean fraud rate per card (training data only)
    if "isFraud" in df_sorted.columns:
        risk = df_sorted.groupby("card1")["isFraud"].mean().rename("merchant_risk_score")
        df_sorted = df_sorted.merge(risk, on="card1", how="left")
    else:
        df_sorted["merchant_risk_score"] = 0.0
    df_sorted["merchant_risk_score"] = df_sorted["merchant_risk_score"].fillna(0.0)

    # distance_from_home — abs deviation from modal addr1, scaled to float
    addr1 = df_sorted["addr1"].fillna(0)
    modal_addr1 = (
        df_sorted.assign(addr1_filled=addr1)
        .groupby("card1")["addr1_filled"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else 0)
        .rename("modal_addr1")
    )
    df_sorted = df_sorted.merge(modal_addr1, on="card1", how="left")
    df_sorted["distance_from_home"] = (addr1 - df_sorted["modal_addr1"]).abs() / 1000.0
    df_sorted.drop(columns=["modal_addr1"], inplace=True)

    # Temporal features
    df_sorted["hour_of_day"] = df_sorted["event_timestamp"].dt.hour.astype("int64")
    df_sorted["is_weekend"] = (df_sorted["event_timestamp"].dt.dayofweek >= 5).astype("int64")

    return df_sorted


def save_features(df: pd.DataFrame, path: str = FEATURES_PARQUET) -> None:
    feature_cols = [
        "TransactionID", "card1", "event_timestamp",
        "tx_amount_last_1h", "tx_count_last_10m", "avg_amount_30d",
        "merchant_risk_score", "distance_from_home",
        "hour_of_day", "is_weekend",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[feature_cols].to_parquet(path, index=False)
    print(f"Saved {len(df):,} feature rows -> {path}")


if __name__ == "__main__":
    print(f"Loading {TRAIN_PARQUET} ...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Computing features for {len(train_df):,} rows ...")
    features_df = build_features(train_df)
    save_features(features_df)
    print("Feature columns:", [c for c in features_df.columns if c not in train_df.columns])
