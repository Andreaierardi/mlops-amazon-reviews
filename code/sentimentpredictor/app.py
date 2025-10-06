import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

from preprocessing.datapreparator import DataPreparator
from preprocessing.transformation import TextTfidfTransformer
# -------- Config --------
MODEL_PATH = Path("artifacts/pipeline.joblib")  # keep it simple & stable
MODEL_PATH = Path(__file__).resolve().parents[2] / "output" / "run_20251005_195650" / "pipeline.joblib"

MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
MAX_SENTENCES = 256
MAX_LEN = 1000  # chars per sentence guardrail


app = FastAPI(title="Amazon Review Sentiment API",
              default_response_class=ORJSONResponse)

class PredictIn(BaseModel):
    sentences: list[str]

class PredictOut(BaseModel):
    sentiments: list[str]

# Load once at startup
pipe = None

@app.on_event("startup")
def _load():
    global pipe
    pipe = joblib.load(MODEL_PATH)
    if pipe is None:
        raise RuntimeError("Model failed to load")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"model_loaded": pipe is not None}

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


