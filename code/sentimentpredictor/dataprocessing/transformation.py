from __future__ import annotations

from typing import Any, Dict, Optional, Union, Iterable
import re, string, html

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TextTfidfTransformer(BaseEstimator, TransformerMixin):
    """
    TF-IDF transformer that operates on a 1-D text sequence (pandas.Series or list of str).

    Config schema (example):
    ------------------------
    cleaning:
      enabled: true
      unescape_html: true
      lowercase: true
      remove_urls: true
      remove_html_tags: true
      remove_digits: true
      remove_punct: true
      squeeze_ws: true
    tfidf:
      lowercase: false         # keep false: lowercasing is done in cleaning
      strip_accents: unicode
      analyzer: word
      ngram_range: [1, 2]
      min_df: 2
      max_features: 200000
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

    def _validate_config(self):
        cfg = self.config
        
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


    # -------------------------- helpers --------------------------
    def _clean_text(self, t: Any) -> str:
        """Apply minimal, fast cleaning steps according to `self.cleaning` flags."""
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

    def _to_series(self, X: Iterable[Any]) -> pd.Series:
        """Ensure we operate on a pandas Series of strings."""
        if isinstance(X, pd.Series):
            s = X
        elif isinstance(X, (list, tuple)):
            s = pd.Series(X)
        else:
            raise TypeError("TextTfidfTransformer expects a pandas.Series or a 1-D list/tuple of strings.")
        return s.astype(str)

    # ------------------------- sklearn API -----------------------
    def fit(self, X, y=None):
        s = self._to_series(X)
        cleaned = s.map(self._clean_text)
        self.vectorizer_ = TfidfVectorizer(**self.tfidf_params).fit(cleaned)
        return self

    def transform(self, X):
        if self.vectorizer_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() or fit_transform() first.")
        s = self._to_series(X)
        cleaned = s.map(self._clean_text)
        return self.vectorizer_.transform(cleaned)


    # Compatibility utilities
    def get_feature_names_out(self, input_features=None):
        """
        Expose learned feature names, e.g., for inspection or export.
        """
        if self.vectorizer_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")
        return self.vectorizer_.get_feature_names_out()
    

#@dataclass
#class DataTransformationConfig:
#    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
#