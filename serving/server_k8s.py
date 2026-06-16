import os
import sys
import logging

_SERVING_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_SERVING_DIR)
_FEATURES_DIR = os.path.join(_PROJECT_DIR, "features")

for _p in (_SERVING_DIR, _FEATURES_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
import redis as redis_lib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from predict import predict as _predict

load_dotenv(os.path.join(_PROJECT_DIR, ".env"))
logging.basicConfig(level=logging.INFO)

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", f"file:{os.path.join(_PROJECT_DIR, 'mlruns')}")
MODEL_NAME  = "fraud_detector"
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()
prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
if not prod:
    raise RuntimeError("No Production model version found in registry")

_model_version = str(prod[0].version)
# MLflow artifact source may be a file:// URI or a bare path; strip the scheme.
_source = prod[0].source
if _source.startswith("file://"):
    _source = _source[len("file://"):]
elif _source.startswith("file:"):
    _source = _source[len("file:"):]
# Remap to /app/mlruns/... only when the host path doesn't exist (i.e. inside container)
_idx = _source.find("/mlruns/")
if _idx >= 0 and not os.path.exists(_source):
    _artifact_path = "/app" + _source[_idx:]
else:
    _artifact_path = _source
_booster_path = os.path.join(_artifact_path, "model.xgb")
logging.info("Loading model v%s from %s", _model_version, _booster_path)
_booster = xgb.Booster()
_booster.load_model(_booster_path)


class _BoosterWrapper:
    """Thin wrapper so predict.py's model.predict_proba(X) works with a raw Booster."""
    def __init__(self, booster: xgb.Booster):
        self._b = booster

    def predict_proba(self, X) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        proba_pos = self._b.predict(dmat)
        return np.column_stack([1 - proba_pos, proba_pos])


_model = _BoosterWrapper(_booster)
logging.info("Model v%s loaded", _model_version)

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


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": _model_version,
        "redis_connected": _redis_ping(),
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = _predict(
            user_id=request.user_id,
            amount=request.amount,
            merchant_id=request.merchant_id,
            timestamp=request.timestamp,
            location=request.location,
            model=_model,
            model_version=_model_version,
        )
        return result
    except Exception as exc:
        logging.error("Prediction failed: %s", exc)
        return JSONResponse(status_code=503, content={"error": "model unavailable"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
