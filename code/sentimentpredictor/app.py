import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

from preprocessing.datapreparator import DataPreparator
from preprocessing.transformation import TextTfidfTransformer
class PredictIn(BaseModel):
    sentences: list[str]

class PredictOut(BaseModel):
    sentiments: list[str]

MODEL_PATH = Path(__file__).resolve().parents[2] / "output" / "run_20251005_195650" / "pipeline.joblib"
mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
app = FastAPI(title="Amazon Review Sentiment API")

@app.on_event("startup")
def _load():
    global pipe
    pipe = joblib.load(MODEL_PATH)  # path in container/host
    assert pipe is not None

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    df = pd.DataFrame({"sentences": inp.sentences})
    labels = pd.Series(pipe.predict(df[["sentences"]])).map(mapping).tolist()

    return {"sentiments": labels}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

