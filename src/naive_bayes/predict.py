"""Prediction interface for Member 6. No fitting occurs in this module."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from preprocessing import normalize_email

DEFAULT_MODEL = Path(__file__).resolve().parent / "artifacts" / "naive_bayes_pipeline.joblib"


def load_model(path=DEFAULT_MODEL):
    return joblib.load(path)


def predict_emails(texts, model=None):
    """Return one {label, label_name, spam_probability, out_of_vocabulary} per email."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if not texts:
        return []
    for text in texts:
        if not normalize_email(text):
            raise ValueError("Email text cannot be empty.")
    if model is None:
        model = load_model()
    matrix = model.named_steps["tfidf"].transform(texts)
    classifier = model.named_steps["nb"]
    labels = classifier.predict(matrix)
    spam_col = int(np.flatnonzero(classifier.classes_ == 1)[0])
    probabilities = classifier.predict_proba(matrix)[:, spam_col]
    return [{"label": int(label), "label_name": "Spam" if label == 1 else "Legitimate",
        "spam_probability": float(probability), "out_of_vocabulary": bool(zero)}
        for label, probability, zero in zip(labels, probabilities, matrix.getnnz(axis=1) == 0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--text-file", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
    try:
        print(json.dumps(predict_emails([text], load_model(args.model))[0], ensure_ascii=False, indent=2))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
