import fastapi
from fastapi.testclient import TestClient
from app import app  # resolves because PYTHONPATH includes code/

client = TestClient(app)



def test_predict_single():
    r = client.post("/predict", json={"text": "Great book!"})
    assert r.status_code == 200
    assert "sentiment" in r.json()