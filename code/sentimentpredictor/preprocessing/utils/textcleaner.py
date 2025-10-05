from sklearn.base import BaseEstimator, TransformerMixin
import re, string, html
import numpy as np


# Minimal cleaner as a transformer (so it’s serialized with the pipeline)
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        def clean(t):
            if not isinstance(t, str): return ""
            t = html.unescape(t).lower()
            t = re.sub(r'http\S+|www\S+', ' ', t)
            t = re.sub(r'<[^>]+>', ' ', t)
            t = re.sub(r'\d+', ' ', t)
            t = t.translate(str.maketrans('', '', string.punctuation))
            t = re.sub(r'\s+', ' ', t).strip()
            return t
        return np.array([clean(x) for x in X])