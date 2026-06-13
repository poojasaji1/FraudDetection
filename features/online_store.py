import os
import sys
import time
import logging

_FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)

from feast import FeatureStore

logger = logging.getLogger(__name__)

_FEATURE_REFS = [
    "transaction_features:tx_amount_last_1h",
    "transaction_features:tx_count_last_10m",
    "transaction_features:avg_amount_30d",
    "transaction_features:merchant_risk_score",
    "transaction_features:distance_from_home",
    "transaction_features:hour_of_day",
    "transaction_features:is_weekend",
]

_DEFAULTS = {
    "tx_amount_last_1h": 0.0,
    "tx_count_last_10m": 0.0,
    "avg_amount_30d": 0.0,
    "merchant_risk_score": 0.0,
    "distance_from_home": 0.0,
    "hour_of_day": 0,
    "is_weekend": 0,
}

_store: FeatureStore | None = None


def _get_store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore(repo_path=_FEATURES_DIR)
    return _store


def get_online_features(card1: int) -> dict:
    """Retrieve latest features for a card from Redis.

    Logs retrieval latency. Raises RuntimeError if latency > 10 ms.
    Returns zero-filled defaults for unknown card1 values.
    """
    store = _get_store()

    t0 = time.perf_counter()
    raw = store.get_online_features(
        features=_FEATURE_REFS,
        entity_rows=[{"card1": card1}],
    ).to_dict()
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info("online feature latency: %.2f ms", latency_ms)
    if latency_ms > 5:
        logger.warning("feature latency %.2f ms exceeded 5 ms target", latency_ms)
    if latency_ms > 10:
        raise RuntimeError(f"Feature retrieval latency {latency_ms:.2f} ms exceeded 10 ms threshold")

    features = {}
    for ref in _FEATURE_REFS:
        feat_name = ref.split(":")[1]
        val = raw.get(feat_name, [None])[0]
        features[feat_name] = val if val is not None else _DEFAULTS[feat_name]

    return features


if __name__ == "__main__":
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    project_dir = os.path.dirname(_FEATURES_DIR)
    df = pd.read_parquet(os.path.join(project_dir, "data", "processed", "features.parquet"))
    sample_card = int(df["card1"].iloc[0])
    print(f"Fetching online features for card1={sample_card}")
    print(get_online_features(sample_card))
