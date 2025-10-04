import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

from service.textcleaner import TextCleaner
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment_pipeline.pkl"
mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

app = FastAPI(title="Amazon Review Sentiment API")

class Review(BaseModel):
    text: str

sentiment_model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict_sentiment(review: Review):
    pred = sentiment_model.predict([review.text])[0]
    return {"sentiment": str(mapping[pred])}   # <- ensure JSON-serializable

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

