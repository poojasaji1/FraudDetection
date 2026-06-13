import os
import pytest
import pandas as pd

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_PARQUET = os.path.join(_PROJECT_DIR, "data", "processed", "features.parquet")

# Skip the whole module if Redis is not reachable
try:
    import redis as _redis_lib
    _r = _redis_lib.Redis(host="localhost", port=6379, socket_connect_timeout=1)
    _r.ping()
    REDIS_UP = True
except Exception:
    REDIS_UP = False

pytestmark = pytest.mark.skipif(not REDIS_UP, reason="Redis not available — run `make services` first")


@pytest.fixture(scope="module")
def store():
    from feast import FeatureStore
    features_dir = os.path.join(_PROJECT_DIR, "features")
    return FeatureStore(repo_path=features_dir)


@pytest.fixture(scope="module")
def card1_samples():
    df = pd.read_parquet(FEATURES_PARQUET)
    # Pick 3 cards that have been materialized (most frequent = most data)
    return [int(v) for v in df["card1"].value_counts().index[:3]]


_FEATURE_NAMES = [
    "tx_amount_last_1h",
    "tx_count_last_10m",
    "avg_amount_30d",
    "merchant_risk_score",
    "distance_from_home",
    "hour_of_day",
    "is_weekend",
]

_FEATURE_REFS = [f"transaction_features:{f}" for f in _FEATURE_NAMES]


def test_offline_online_parity(store, card1_samples):
    """Features returned by the online store (Redis) must match the latest row
    for each card1 in features.parquet within tolerance 1e-3."""
    feats_df = pd.read_parquet(FEATURES_PARQUET)

    for card1 in card1_samples:
        card_rows = feats_df[feats_df["card1"] == card1]
        assert not card_rows.empty, f"card1={card1} not in features.parquet"
        offline_row = card_rows.loc[card_rows["event_timestamp"].idxmax()]

        online_raw = store.get_online_features(
            features=_FEATURE_REFS,
            entity_rows=[{"card1": card1}],
        ).to_dict()

        for feat in _FEATURE_NAMES:
            online_val = online_raw.get(feat, [None])[0]
            offline_val = offline_row[feat]

            assert online_val is not None, f"card1={card1} feat={feat} missing from online store"
            assert abs(float(online_val) - float(offline_val)) <= 1e-3, (
                f"card1={card1} feat={feat}: offline={offline_val} online={online_val}"
            )
