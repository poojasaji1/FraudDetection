import os
import sys
import pandas as pd
import pyarrow.parquet as _pq

_FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)

_PROJECT_DIR = os.path.dirname(_FEATURES_DIR)
FEATURES_PARQUET = os.path.join(_PROJECT_DIR, "data", "processed", "features.parquet")

FEATURE_NAMES = [
    "tx_amount_last_1h",
    "tx_count_last_10m",
    "avg_amount_30d",
    "merchant_risk_score",
    "distance_from_home",
    "hour_of_day",
    "is_weekend",
]


def get_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join pre-computed point-in-time features onto a transaction DataFrame.

    Merges on TransactionID (exact 1:1) so there are no duplicate-timestamp
    alignment issues. features.parquet was computed with PIT correctness in
    the feature engineering step (Day 2).
    """
    # Use pyarrow directly to avoid pandas-cached-filesystem issues with older parquet files
    features_df = _pq.read_table(FEATURES_PARQUET).to_pandas()[["TransactionID"] + FEATURE_NAMES]
    return df.merge(features_df, on="TransactionID", how="inner")


if __name__ == "__main__":
    train_path = os.path.join(_PROJECT_DIR, "data", "processed", "train.parquet")
    df = pd.read_parquet(train_path).head(1000)
    result = get_training_features(df)
    print(result[["TransactionID"] + FEATURE_NAMES].head())
    print("Shape:", result.shape)
