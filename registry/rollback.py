import os
import time

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

_REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_REGISTRY_DIR)

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def rollback(model_name: str = "fraud_detector") -> bool:
    """Swap the current Production version to Archived and promote the latest Archived to Production.

    Asserts completion within 30 seconds.
    """
    t0 = time.time()
    client = MlflowClient()

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        print(f"No Production version found for '{model_name}'")
        return False

    archived_versions = client.get_latest_versions(model_name, stages=["Archived"])
    if not archived_versions:
        print(f"No Archived version to roll back to for '{model_name}'")
        return False

    prod_v = prod_versions[0]
    # Most recent archived = highest version number
    prev_v = max(archived_versions, key=lambda v: int(v.version))

    client.transition_model_version_stage(model_name, prod_v.version, "Archived")
    client.transition_model_version_stage(model_name, prev_v.version, "Production")

    elapsed = time.time() - t0
    assert elapsed < 30, f"Rollback took {elapsed:.1f}s — exceeded 30 s limit"
    print(f"Rolled back to version {prev_v.version}  ({elapsed:.2f}s)")
    return True


if __name__ == "__main__":
    rollback()
