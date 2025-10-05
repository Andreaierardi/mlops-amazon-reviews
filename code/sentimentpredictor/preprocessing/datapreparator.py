from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union

# NLTK sentence splitter
import nltk
from nltk.tokenize import sent_tokenize

class DataPreparator:
    """
    Simple data transformer that only performs `transform()`.
    Responsibilities:
      - Create the target column if missing, using 'rating' and a mapping defined in the config.
      - Apply basic DataFrame filters (dropna, min text length, deduplication, column selection).

    Minimal config example:
    columns:
      text: text
      rating: rating
      label: sentiment
    labeling:
      mapping: {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
      dtype: int
    filters:
      dropna: true
      keep_only_text_and_label: false
      min_text_len: 1
      explode_sentences: true
    """


    def __init__(self, config: Dict[str, Any]):
        # config: path to YAML or a dict. If 'section' is provided, use config[section]
        self.config = config
        self.text_col: str = "text"
        self.rating_col: Optional[str] = "rating"
        self.label_col: str = "sentiment"
        self._validate_config()

    # ------------------------- config handling -------------------------
    def _validate_config(self):
        cfg = self.config

        cols = cfg["columns"]
        self.text_col = cols.get("text", "text")
        self.rating_col = cols.get("rating", "rating")
        self.label_col = cols.get("label", "sentiment")
        self.sentence_col = cols.get("sentence", "sentence")

        # derived settings
        labeling = cfg["labeling"]
        self.mapping = labeling.get("mapping")
        self.label_dtype = labeling.get("dtype", int)

        filters = cfg.get("filters", {})
        self.dropna = filters.get("dropna", False)
        self.min_text_len = filters.get("min_text_len", None)
        self.keep_only_sentence_and_label = filters.get("keep_only_sentence_and_label", True)
        self.explode_sentences = filters.get("explode_sentences", False)

    @staticmethod
    def _split_into_sentences(text: Any) -> list[str]:
        """Safe sentence split with fallback."""
        try:
            t = "" if text is None else str(text)
            return sent_tokenize(t)
        except Exception:
            return ["" if text is None else str(text)]
        
    # ------------------------- transform -------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # crea label se assente
        if self.label_col not in df.columns:
            if self.mapping is None:
                raise ValueError("Missing 'labeling.mapping' in config to derive the label.")
            if self.rating_col not in df.columns:
                raise ValueError(f"Column '{self.rating_col}' not found to derive the label.")
            df[self.label_col] = df[self.rating_col].map(
                lambda v: self.mapping.get(int(v), np.nan) if pd.notna(v) else np.nan
            )

        # cast dtype
        df[self.label_col] = df[self.label_col].astype(self.label_dtype, errors="ignore")

        # Basic filters
        if self.dropna:
            df = df.dropna(subset=[self.text_col, self.label_col])
        if isinstance(self.min_text_len, int) and self.min_text_len > 0:
            df = df[df[self.text_col].astype(str).str.len() >= self.min_text_len]

        # Sentence explosion (optional)
        if self.explode_sentences:
            df[self.sentence_col] = df[self.text_col].apply(self._split_into_sentences)
            df = df.explode(self.sentence_col, ignore_index=True)
            df = df.rename(columns={self.sentence_col: self.sentence_col})
        else:
            # if no explosion, keep original text as "sentence" column
            df[self.sentence_col] = df[self.text_col]

        # Keep only necessary columns
        if self.keep_only_sentence_and_label:
            df = df[[self.sentence_col, self.label_col]]

        # Enforce label dtype
        df[self.label_col] = df[self.label_col].astype(self.label_dtype, errors="ignore")

        return df

