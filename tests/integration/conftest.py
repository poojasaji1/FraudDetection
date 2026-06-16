"""Session-scoped setup for integration tests."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FEATURES_DIR = os.path.join(_PROJECT_DIR, "features")

for _p in (_PROJECT_DIR, _FEATURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest


@pytest.fixture(scope="session", autouse=True)
def feast_apply_fixture():
    """Register Feast feature views using the Python API (avoids repo-scan timeout)."""
    try:
        from feast import FeatureStore, Entity, FeatureView, FileSource, Field
        from feast.types import Float64, Int64
        from datetime import timedelta

        _features_parquet = os.path.join(
            _PROJECT_DIR, "data", "processed", "features.parquet"
        )
        card = Entity(name="card", join_keys=["card1"])
        source = FileSource(path=_features_parquet, timestamp_field="event_timestamp")
        fv = FeatureView(
            name="transaction_features",
            entities=[card],
            ttl=timedelta(days=90),
            source=source,
            schema=[
                Field(name="tx_amount_last_1h",  dtype=Float64),
                Field(name="tx_count_last_10m",  dtype=Float64),
                Field(name="avg_amount_30d",      dtype=Float64),
                Field(name="merchant_risk_score", dtype=Float64),
                Field(name="distance_from_home",  dtype=Float64),
                Field(name="hour_of_day",         dtype=Int64),
                Field(name="is_weekend",          dtype=Int64),
            ],
        )
        store = FeatureStore(repo_path=_FEATURES_DIR)
        store.apply([card, fv])
    except Exception as exc:
        print(f"\n[conftest] feast apply warning: {exc}")
