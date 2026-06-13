import os
import sys
import logging

_SERVING_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_SERVING_DIR)
_FEATURES_DIR = os.path.join(_PROJECT_DIR, "features")

for _p in (_SERVING_DIR, _FEATURES_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import redis as redis_lib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ray import serve
from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

from predict import predict as _predict

logging.basicConfig(level=logging.INFO)

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", f"file:{os.path.join(_PROJECT_DIR, 'mlruns')}")
MODEL_NAME  = "fraud_detector"
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))

app = FastAPI()


class PredictRequest(BaseModel):
    user_id:     str
    amount:      float = Field(gt=0)
    merchant_id: str
    timestamp:   str
    location:    dict


class PredictResponse(BaseModel):
    fraud_score:   float
    label:         str
    latency_ms:    float
    model_version: str


def _redis_ping() -> bool:
    try:
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        return True
    except Exception:
        return False


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.5},
)
@serve.ingress(app)
class FraudDetector:

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        self.model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/Production")

        client = MlflowClient()
        prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        self.model_version = str(prod[0].version) if prod else "unknown"

        self._redis_ok = _redis_ping()
        if not self._redis_ok:
            logging.warning("Redis unavailable at startup — online features will use defaults")

        print(f"FraudDetector loaded model version {self.model_version}")
        logging.info("FraudDetector loaded model version %s", self.model_version)

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest):
        try:
            result = _predict(
                user_id=request.user_id,
                amount=request.amount,
                merchant_id=request.merchant_id,
                timestamp=request.timestamp,
                location=request.location,
                model=self.model,
                model_version=self.model_version,
            )
            return result
        except Exception as exc:
            logging.error("Model prediction failed: %s", exc)
            return JSONResponse(
                status_code=503,
                content={"error": "model unavailable"},
            )

    @app.get("/health")
    async def health(self):
        return {
            "status": "healthy",
            "model_version": self.model_version,
            "redis_connected": _redis_ping(),
        }


if __name__ == "__main__":
    import time
    serve.run(FraudDetector.bind(), route_prefix="/")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        serve.shutdown()
