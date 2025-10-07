import numpy as np
import scipy.sparse as sp
from sentimentpredictor.preprocessing.transformation import TextTfidfTransformer
import yaml
from pathlib import Path

def test_tfidf_transformer_fit_transform():
    texts = [
        "Amazing quality and service",
        "Service was okay",
        "Terrible experience",
        "Amazing amazing amazing"
    ]
    cfg = yaml.safe_load(Path('code/sentimentpredictor/config/transformation_config.yaml').read_text())

    tf = TextTfidfTransformer(cfg)

    X = tf.fit_transform(texts)
    # Should be a sparse matrix with rows = len(texts)
    assert sp.issparse(X)
    assert X.shape[0] == len(texts)

    # Transform again (idempotent)
    X2 = tf.transform(texts)
    assert X2.shape == X.shape

    # Feature names available after fit
    names = tf.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.size > 0