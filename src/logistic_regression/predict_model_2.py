"""Predict a new email with the trained Member 3 model.
Usage: python predict_model_2.py --text "Subject: meeting Please review the notes"
"""
import argparse
import json
from pathlib import Path
import joblib
from preprocessing import normalize_email

DEFAULT_PATH = Path(__file__).resolve().parent / 'artifacts' / 'logistic_regression_pipeline.joblib'


def predict_emails(texts, model=None):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    if not all(isinstance(t, str) and normalize_email(t) for t in texts):
        raise ValueError('Email input must be a non-empty string.')
    if model is None:
        model = joblib.load(DEFAULT_PATH)
    x = model.named_steps['tfidf'].transform(texts)
    pred = model.named_steps['lr'].predict(x)
    prob = model.named_steps['lr'].predict_proba(x)[:, 1]
    return [{'label': int(v), 'label_name': 'Spam' if v == 1 else 'Legitimate',
             'spam_probability': float(p), 'out_of_vocabulary': bool(z)}
            for v, p, z in zip(pred, prob, x.getnnz(axis=1) == 0)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_PATH)
    a = parser.parse_args()
    print(json.dumps(predict_emails(a.text, joblib.load(a.model))[0], indent=2))
