"""Shared, deterministic email preprocessing for Members 2 and 3.

Only the TF-IDF vocabulary and IDF weights are learned; fit them on training
emails only. Keep this module available when loading the saved joblib pipeline.
"""
import ast
import html
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def decode_email(text: str) -> str:
    """Accept ordinary pasted text or the dataset's serialized list of strings."""
    if not isinstance(text, str):
        raise TypeError("Email text must be a string.")
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, RecursionError):
            value = None
        if isinstance(value, list) and all(isinstance(part, str) for part in value):
            text = "".join(value)
    return text


def normalize_email(text: str) -> str:
    """Decode the wrapper, HTML entities, case and whitespace without fitting."""
    text = html.unescape(decode_email(text))
    return re.sub(r"\s+", " ", text).strip().lower()


def make_vectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=False,
                    max_features=50000):
    """The common starting configuration for a controlled model comparison.

Preserve subject and body; no stopword list or stemming. The word tokenizer
retains Unicode word tokens of length >= 2, including numeric tokens. Training
learns vocabulary and smoothed IDF; each document receives L2 normalization.
"""
    return TfidfVectorizer(
        preprocessor=normalize_email,
        lowercase=False,  # normalize_email explicitly performs lowercasing
        strip_accents=None,
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
        stop_words=None,
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_df=1.0,
        max_features=max_features,
        binary=False,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=sublinear_tf,
        norm="l2",
    )


def load_split(path):
    """Validate a supplied split without silently dropping/relabeling rows."""
    path = Path(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if list(frame.columns) != ["text", "label"]:
        raise ValueError(f"{path.name}: expected columns text,label")
    if not frame["text"].map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"{path.name}: non-string email")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{path.name}: label must be 0 (ham) or 1 (spam)")
    if set(frame["label"]) != {0, 1}:
        raise ValueError(f"{path.name}: both classes are required")
    if frame["text"].map(normalize_email).eq("").any():
        raise ValueError(f"{path.name}: empty email after normalization")
    frame["label"] = frame["label"].astype(int)
    return frame
