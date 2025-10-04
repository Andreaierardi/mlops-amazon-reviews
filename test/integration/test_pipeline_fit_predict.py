import joblib
from pathlib import Path

def test_saved_model_can_predict():
    pkl = Path("models/sentiment_pipeline.pkl")
    assert pkl.exists()
    pipe = joblib.load(pkl)
    out = pipe.predict(["This was amazing!", "Not good", "It was okay."])
    assert len(out) == 3