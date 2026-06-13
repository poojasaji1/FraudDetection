import os
import sys
from datetime import datetime, timezone

_FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
if _FEATURES_DIR not in sys.path:
    sys.path.insert(0, _FEATURES_DIR)

from feast import FeatureStore


def materialize() -> None:
    store = FeatureStore(repo_path=_FEATURES_DIR)
    end_date = datetime.now(tz=timezone.utc)
    print(f"Materializing features up to {end_date.isoformat()} ...")
    store.materialize_incremental(end_date=end_date)
    print("Materialization complete.")


if __name__ == "__main__":
    materialize()
