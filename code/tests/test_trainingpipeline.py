from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from unittest import mock

# Import the training pipeline module
import sentimentpredictor.train.training_pipeline as tp
from sentimentpredictor.preprocessing.transformation import TextTfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ============================================================
# Tests
# ============================================================
def test_prepare_data(monkeypatch):
    """Ensure prepare_data runs and drops NaN."""
    df = pd.DataFrame({
        "sentence": ["a", "b", None],
        "sentiment": [1, 2, 3]
    })
    cfg = {"dropna": True}

    class DummyPrep:
        def __init__(self, cfg): pass
        def transform(self, df): return df

    monkeypatch.setattr(tp, "DataPreparator", DummyPrep)
    result = tp.prepare_data(df, cfg, "sentence", "sentiment")
    assert "sentence" in result.columns
    assert len(result) == 2  # NaN dropped


def test_evaluate_model_outputs():
    """Validate structure of metrics returned by evaluate_model."""
    X_val = pd.Series(["good", "bad"])
    y_val = pd.Series([1, 0])
    y_train = pd.Series([1, 0])

    class DummyModel:
        def predict(self, x): return [1, 0]

    pipe = DummyModel()
    metrics, cm, report = tp.evaluate_model(pipe, X_val, y_val, y_train)
    assert "accuracy" in metrics
    assert isinstance(cm, np.ndarray)
    assert isinstance(report, dict)

def test_build_pipeline():
    """Validate pipeline build"""
    transf_config = yaml.safe_load(Path('code/sentimentpredictor/config/transformation_config.yaml').read_text())
    training_config = yaml.safe_load(Path('code/sentimentpredictor/config/train_config.yaml').read_text())

    p = tp.build_pipeline(transf_config, training_config['model'])


    assert isinstance(p, Pipeline), "Should return sklearn.pipeline.Pipeline"

    step_names = dict(p.named_steps)
    assert "tfidf" in step_names, "Missing TF-IDF transformer step"
    assert "clf" in step_names, "Missing classifier step"

    assert isinstance(p.named_steps["clf"], LogisticRegression)

    clf = p.named_steps["clf"]
    assert clf.max_iter == training_config["model"].get("max_iter", 100), "max_iter not set correctly"

    X = ["good quality", "bad item"]*100
    y = [1, 0] *100
    p.fit(X, y)
    preds = p.predict(X)
    assert len(preds) == len(X)
