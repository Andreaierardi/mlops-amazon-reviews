from __future__ import annotations

from typing import Any, Dict, Optional, Union, Iterable
import re, string, html
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ============================================================
# Logging setup
# ============================================================
logger = logging.getLogger("text-tfidf-transformer")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    
class TextTfidfTransformer(BaseEstimator, TransformerMixin):
    """
    ColumnTransformer-friendly text cleaner + TF-IDF.

    - Works with ColumnTransformer where you pass the column selector separately:
        ColumnTransformer([("text", TextTfidfTransformer(cfg), "sentence"), ...])
      so this class never needs to know the column name.
    - Accepts 1-D text (Series/list/ndarray) or 2-D with a single column.

    """

    def __init__(self, config: Union[str, Dict[str, Any]]):
        """
        Parameters
        ----------
        config : str | dict
            Path to YAML configuration file or a loaded dict.
        """
        self.config = config

        # Populated in _load_config()
        self.cleaning: Dict[str, bool] = {}
        self.tfidf_params: Dict[str, Any] = {}
        self.vectorizer_: Optional[TfidfVectorizer] = None

        self._validate_config()
        logger.info("Initialized TextTfidfTransformer with config keys: %s", list(config.keys()))

    def _validate_config(self):
        cfg = self.config
        logger.debug("Validating TF-IDF transformer configuration...")

        defaults_cleaning = dict(
            enabled=True,
            unescape_html=True,
            lowercase=True,
            remove_urls=True,
            remove_html_tags=True,
            remove_digits=True,
            remove_punct=True,
            squeeze_ws=True,
        )
        self.cleaning = {**defaults_cleaning, **(cfg.get("cleaning") or {})}

        # tfidf vectorizer parameters
        defaults_tfidf = dict(
            lowercase=False,          # already done in cleaning
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=200_000,
        )
        user_tfidf = cfg.get("tfidf") or {}

        # Convert list to tuple for ngram_range if provided as YAML array
        if isinstance(user_tfidf.get("ngram_range"), list):
            user_tfidf["ngram_range"] = tuple(user_tfidf["ngram_range"])
        self.tfidf_params = {**defaults_tfidf, **user_tfidf}

        logger.info(
            "TF-IDF config validated — ngram_range=%s, min_df=%s, max_features=%s",
            self.tfidf_params.get("ngram_range"),
            self.tfidf_params.get("min_df"),
            self.tfidf_params.get("max_features"),
        )

    # --------------------- helpers ---------------------
    def _clean_one(self, t: Any) -> str:
        if not self.cleaning.get("enabled", True):
            return t if isinstance(t, str) else ""
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        if self.cleaning.get("unescape_html", True):
            t = html.unescape(t)
        if self.cleaning.get("lowercase", True):
            t = t.lower()
        if self.cleaning.get("remove_urls", True):
            t = re.sub(r"http\S+|www\S+", " ", t)
        if self.cleaning.get("remove_html_tags", True):
            t = re.sub(r"<[^>]+>", " ", t)
        if self.cleaning.get("remove_digits", True):
            t = re.sub(r"\d+", " ", t)
        if self.cleaning.get("remove_punct", True):
            t = t.translate(str.maketrans("", "", string.punctuation))
        if self.cleaning.get("squeeze_ws", True):
            t = re.sub(r"\s+", " ", t).strip()
        return t


    def _as_series_1d(self, X: Any) -> pd.Series:
        """
        Accept a DataFrame/Series/list/tuple/ndarray and return a 1-D Series[str].
        - If X is 2-D (n, 1), squeeze it.
        - If X is a DataFrame with 1 column, squeeze to Series.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("Expected a single text column (got shape %s)." % (X.shape,))
            X = X.iloc[:, 0]
        if isinstance(X, pd.Series):
            s = X
        elif isinstance(X, (list, tuple)):
            s = pd.Series(X)
        elif isinstance(X, np.ndarray):
            if X.ndim == 2 and X.shape[1] == 1:
                s = pd.Series(X.ravel())
            elif X.ndim == 1:
                s = pd.Series(X)
            else:
                raise ValueError("NumPy input must be 1-D or 2-D with a single column.")
        else:
            raise TypeError("Unsupported input type. Provide DataFrame/Series/list/tuple/ndarray.")
        return s.astype(str)

    # --------------------- sklearn API ---------------------
    def fit(self, X, y=None):
        logger.info("Fitting TF-IDF transformer on input data...")
        s = self._as_series_1d(X).map(self._clean_one)
        self.vectorizer_ = TfidfVectorizer(**self.tfidf_params).fit(s)
        return self

    def transform(self, X):
        logger.info("Transforming text data with TF-IDF...")
        if self.vectorizer_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() or fit_transform() first.")
        s = self._as_series_1d(X).map(self._clean_one)
        return self.vectorizer_.transform(s)

    def get_feature_names_out(self, input_features=None):
        if self.vectorizer_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        return self.vectorizer_.get_feature_names_out()