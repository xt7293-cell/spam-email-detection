"""CSCI323 Member 3 - TF-IDF + Logistic Regression email classifier.

Only train.csv and validation.csv are read. Never open test.csv in this script.
The text processor is copied unchanged from Member 2, and its chosen TF-IDF
configuration is used for a controlled comparison (the same features).

Usage:
    python train_model_2.py --data-dir data
    python train_model_2.py --data-dir path/to/NaiveBayes/data
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from preprocessing import load_split, make_vectorizer, normalize_email

ROOT = Path(__file__).resolve().parent
TFIDF_CONFIG = dict(ngram_range=(1, 1), min_df=2,
                    sublinear_tf=False, max_features=50000)
C_VALUES = (0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, 'balanced')
SEED = 42


def score(y_true, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y_true, pred)),
                precision=float(precision_score(y_true, pred, zero_division=0)),
                recall=float(recall_score(y_true, pred, zero_division=0)),
                f1=float(f1_score(y_true, pred, zero_division=0)),
                fpr=float(fp / (tn + fp)),
                fnr=float(fn / (fn + tp)),
                roc_auc=float(roc_auc_score(y_true, prob)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1] / 'data' / 'splits')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    out = args.output_dir
    start = time.perf_counter()

    train = load_split(args.data_dir / 'train.csv')
    val = load_split(args.data_dir / 'validation.csv')
    print(f'Training: {len(train)} emails; validation: {len(val)} emails.')
    print('TEST SET NOT OPENED. Labels: 0=Legitimate; 1=Spam.')

    # Check normalized train-validation text overlap without modifying
    # the group's agreed data splits.
    tr_text = train.text.map(normalize_email)
    val_text = val.text.map(normalize_email)
    shared = set(tr_text) & set(val_text)
    overlap = val_text.isin(shared).to_numpy()
    print(f'Normalized train/validation overlap: {int(overlap.sum())} validation row(s).')
    if overlap.any():
        print('WARNING: Tell Member 1/5 to resolve the duplicate issue across ALL models.')

    # Refit Member 2's exact selected vectorizer settings on TRAIN ONLY.
    # We do not use Member 1's experimental all-data TF-IDF matrix.
    vectorizer = make_vectorizer(**TFIDF_CONFIG)
    transform_start = time.perf_counter()
    X_train = vectorizer.fit_transform(train.text)
    X_val = vectorizer.transform(val.text)
    featurization_sec = time.perf_counter() - transform_start
    print(f'TF-IDF matrix: train={X_train.shape}, validation={X_val.shape}')
    print(f'Vocabulary size: {len(vectorizer.vocabulary_)}')

    rows = []
    candidates = []
    experiment_id = 0
    for c in C_VALUES:
        for weight in CLASS_WEIGHTS:
            experiment_id += 1
            model = LogisticRegression(C=c, class_weight=weight,
                    solver='liblinear', max_iter=2000,
                    random_state=SEED)
            t0 = time.perf_counter()
            model.fit(X_train, train.label)
            fit_sec = time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = model.predict(X_val)
            prob = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
            infer_sec = time.perf_counter() - t1
            metrics = score(val.label, pred, prob)
            row = dict(experiment_id=experiment_id, C=c,
                       class_weight='none' if weight is None else 'balanced',
                       solver='liblinear', threshold=0.5, max_iter=2000,
                       iterations=int(model.n_iter_.max()),
                       fit_seconds=round(fit_sec, 5),
                       validation_inference_seconds=round(infer_sec, 5),
                       **metrics)
            rows.append(row)
            candidates.append((model, pred, prob))
            print(f'#{experiment_id} C={c}, weight={row["class_weight"]}: '
                  f'F1={metrics["f1"]:.4f}, precision={metrics["precision"]:.4f}, '
                  f'recall={metrics["recall"]:.4f}, FP={metrics["fp"]}, FN={metrics["fn"]}')

    results = pd.DataFrame(rows)
    results.to_csv(out / 'experiments.csv', index=False)
    # Primary model-selection criterion: spam F1, then precision, then earlier ID.
    chosen = sorted(range(len(rows)),
             key=lambda i: (-rows[i]['f1'], -rows[i]['precision'], rows[i]['experiment_id']))[0]
    best_row = rows[chosen]
    best_model, pred, prob = candidates[chosen]

    # The fitted TF-IDF is bundled with Logistic Regression so Member 6 can
    # predict from raw emails without mistakenly fitting on new examples.
    pipeline = Pipeline([('tfidf', vectorizer), ('lr', best_model)])
    model_path = out / 'logistic_regression_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)
    assert np.array_equal(loaded.predict(val.text), pred)
    assert np.allclose(loaded.predict_proba(val.text)[:, 1], prob)
    print('Saved and reloaded pipeline predictions match.')

    predictions = pd.DataFrame({'row_index_0based': np.arange(len(val)),
                 'actual_label': val.label, 'predicted_label': pred,
                 'spam_probability_estimate': prob})
    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    wrong = predictions[predictions.actual_label != predictions.predicted_label].copy()
    wrong['email_excerpt'] = val.iloc[wrong.row_index_0based].text.map(normalize_email).str.slice(0, 350).to_numpy()
    wrong['error_type'] = np.where(wrong.actual_label == 0, 'False Positive', 'False Negative')
    wrong.to_csv(out / 'validation_errors.csv', index=False)

    # Coefficients are associations conditional on the other features, not
    # individual word causal effects.
    feature_names = vectorizer.get_feature_names_out()
    coefficients = best_model.coef_[0]
    top_pos = np.argsort(coefficients)[-20:][::-1]
    top_neg = np.argsort(coefficients)[:20]
    pd.DataFrame({'feature': np.concatenate([feature_names[top_pos], feature_names[top_neg]]),
                  'coefficient': np.concatenate([coefficients[top_pos], coefficients[top_neg]]),
                  'association': ['Spam'] * 20 + ['Legitimate'] * 20}).to_csv(
                      out / 'influential_features.csv', index=False)

    cm = confusion_matrix(val.label, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4.5), layout='constrained')
    ax.imshow(cm, cmap='Blues')
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, str(n), ha='center', va='center',
                color='white' if n > cm.max() / 2 else 'black', fontsize=16)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Legitimate', 'Spam'],
           yticklabels=['Legitimate', 'Spam'], xlabel='Predicted', ylabel='Actual',
           title='Logistic Regression: Validation Confusion Matrix')
    fig.savefig(out / 'validation_confusion_matrix.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for weight, group in results.groupby('class_weight'):
        group = group.sort_values('C')
        ax.plot(group.C, group.f1 * 100, marker='o', label=f'class_weight={weight}')
    ax.set(xscale='log', xlabel='Regularization parameter C (log scale)',
           ylabel='Validation spam F1 (%)', title='Logistic Regression Tuning')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(out / 'validation_tuning.png', dpi=150)
    plt.close(fig)

    # Sensitivity check on the one identified duplicate only, not a replacement
    # for group-wide deduplication and re-splitting.
    no_overlap = None
    if overlap.any():
        no_overlap = score(val.label.to_numpy()[~overlap], pred[~overlap], prob[~overlap])

    summary = dict(method='TF-IDF + Logistic Regression',
        data=dict(training=len(train), validation=len(val), positive_label=1),
        evaluation_split='validation', test_evaluated=False,
        tfidf_config=dict(ngram_range=list(TFIDF_CONFIG['ngram_range']),
                          min_df=TFIDF_CONFIG['min_df'],
                          sublinear_tf=TFIDF_CONFIG['sublinear_tf'],
                          max_features=TFIDF_CONFIG['max_features']),
        feature_fit_scope='train only',
        feature_note='Same TF-IDF settings as Member 2 selected Naive Bayes; reevaluated only on validation.',
        vocabulary_size=len(vectorizer.vocabulary_),
        num_experiments=len(rows),
        selection_rule='highest validation spam F1; then precision; then lower experiment ID',
        selected=best_row,
        baseline_next_to_member2=next(r for r in rows if r['C'] == 1.0 and r['class_weight'] == 'none'),
        overlap_validation_rows=int(overlap.sum()),
        validation_without_normalized_train_overlap=no_overlap,
        featurization_seconds=featurization_sec,
        total_run_seconds=time.perf_counter() - start,
        environment=dict(python=sys.version.split()[0], sklearn=sklearn.__version__,
                         pandas=pd.__version__, platform=platform.platform()),
        probability_note='Uncalibrated model estimates, not guaranteed real-world probabilities.')
    save_json(out / 'validation_results.json', summary)
    print('\nBEST CONFIGURATION:', json.dumps(best_row, indent=2))
    print('Completed. Results saved under:', out)

if __name__ == '__main__':
    main()
