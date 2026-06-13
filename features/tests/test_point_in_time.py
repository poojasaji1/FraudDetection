import os
import numpy as np
import pandas as pd
import pytest

from feature_views import build_features

_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
TRAIN_PARQUET = os.path.normpath(os.path.join(_PROJECT_DIR, "data", "processed", "train.parquet"))


@pytest.fixture(scope="module")
def card_subset():
    df = pd.read_parquet(TRAIN_PARQUET)
    # Pick a card with enough transactions for the rolling test
    card1_val = int(df["card1"].value_counts().index[0])
    rows = df[df["card1"] == card1_val].sort_values("TransactionDT").head(5).copy()
    return rows, card1_val


def test_no_future_leakage(card_subset):
    """Adding a transaction 1 hour in the future must not change tx_amount_last_1h
    for the original rows — verifies closed='left' rolling window is correct."""
    rows, card1_val = card_subset

    # Baseline features for the 5 original rows
    base = build_features(rows)
    base_amounts = base["tx_amount_last_1h"].values.copy()

    # Inject a future row: 1 h + 1 s after the last transaction, extreme amount
    last_dt = int(rows["TransactionDT"].max())
    future_row = rows.iloc[[-1]].copy()
    future_row["TransactionDT"] = last_dt + 3601
    future_row["TransactionAmt"] = 999_999.0

    combined = pd.concat([rows, future_row], ignore_index=True)
    with_future = build_features(combined)

    # Original 5 rows are the first 5 after sorting by (card1, event_timestamp)
    original_feats = with_future.iloc[:5]

    np.testing.assert_allclose(
        base_amounts,
        original_feats["tx_amount_last_1h"].values,
        rtol=1e-5,
        err_msg="tx_amount_last_1h changed after injecting a future transaction — leakage detected",
    )
