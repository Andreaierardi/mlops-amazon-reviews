import sys
from pathlib import Path
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

from preprocessing.datapreparator import DataPreparator
from preprocessing.transformation import TextTfidfTransformer

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("sentiment-api")


app = FastAPI(title="Amazon Review Sentiment API",
              default_response_class=ORJSONResponse)

# ---------------------------------------------------------
# Model loading (MLflow first, then local fallback)
# ---------------------------------------------------------
def load_model():
    """
    Load model from MLflow if MODEL_URI and MLFLOW_TRACKING_URI are set.
    Fallback to local artifacts/pipeline.joblib if MLflow loading fails
    or vars are not provided.
    """
    model_source = "local"
    model_obj = None
    
    logger.info(f'Loading ENV vars',os.getenv("MODEL_URI"), os.getenv("MLFLOW_TRACKING_URI")  )
    model_uri = os.getenv("MODEL_URI")                  
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")      # e.g., http://host.docker.internal:5000

    # Try MLflow first (only if both env vars are present)
    if model_uri and tracking_uri:
        try:
            import mlflow
            import mlflow.pyfunc
            mlflow.set_tracking_uri(tracking_uri)

            client = mlflow.MlflowClient()

            logger.info(f"Attempting MLflow model load: {model_uri}")
            model_obj = mlflow.sklearn.load_model(model_uri) #models:/sentiment-predictor/latest'
            model_source = "mlflow"
            logger.info("Loaded model from MLflow.")
        except Exception as e:
            logger.warning(f"MLflow load failed ({e}). Falling back to local artifact...")

    return model_obj, model_source


pipe, MODEL_SOURCE = load_model()
logger.info(f"✅ Model loaded from: {MODEL_SOURCE}")

# -------- Config --------
MODEL_PATH = Path("artifacts/pipeline.joblib")  # keep it simple & stable
MODEL_PATH = Path(__file__).resolve().parents[2] / "output" / "run_20251005_195650" / "pipeline.joblib"

MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
MAX_SENTENCES = 256
MAX_LEN = 1000  # chars per sentence guardrail

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class PredictIn(BaseModel):
    sentences: list[str]

class PredictOut(BaseModel):
    sentiments: list[str]

# ---------------------------------------------------------
# Inference logic
# ---------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_source": MODEL_SOURCE}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    # light, fast validation (avoid heavy pydantic logic on huge payloads)
    n = len(inp.sentences)
    if n == 0:
        raise HTTPException(422, "Empty sentences")
    if n > MAX_SENTENCES:
        raise HTTPException(413, f"Too many sentences (>{MAX_SENTENCES})")
    if any((s is None) or (len(s) > MAX_LEN) for s in inp.sentences):
        raise HTTPException(413, f"Sentence too long (>{MAX_LEN} chars)")

    preds = pipe.predict(inp.sentences)
    sentiments = [MAPPING[int(p)] for p in preds]
    return PredictOut(sentiments=sentiments)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, 
                host="0.0.0.0", 
                port=8000,
                workers=4,
                loop="uvloop",
                http="httptools",
                timeout_keep_alive=5,
                reload=False,
                access_log=False,
        )


