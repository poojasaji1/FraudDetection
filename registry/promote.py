import os
import sys

_REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_REGISTRY_DIR)

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def promote_to_production(model_name: str = "fraud_detector") -> bool:
    """Transition the latest Staging version to Production, archiving the current Production."""
    client = MlflowClient()

    staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging:
        print(f"No Staging version found for '{model_name}'")
        return False

    staging_v = staging[0]

    # Archive current Production versions
    for pv in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(model_name, pv.version, "Archived")
        print(f"Archived version {pv.version}")

    client.transition_model_version_stage(model_name, staging_v.version, "Production")
    print(f"Promoted version {staging_v.version} to Production")
    return True


if __name__ == "__main__":
    promote_to_production()
