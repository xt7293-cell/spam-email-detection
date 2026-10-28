"""Train and select a MultinomialNB email classifier on shared train/validation.

Never reads test.csv. The 32 settings and ranking rule are fixed in this file.
Run: python train_model_1.py --data-dir data --output-dir artifacts
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep plotting caches inside this project rather than the user's home folder.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import make_vectorizer, load_split, normalize_email

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics(y, prediction, probability=None):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
    }
    if probability is not None:
        result["roc_auc"] = float(roc_auc_score(y, probability))
        result["average_precision"] = float(average_precision_score(y, probability))
    return result


def audit_data(train, validation, data_dir, out):
    keys = {}
    audit = {"scope": "train and validation only; test.csv is not opened",
             "splits_preserved": True, "splits": {}}
    for name, frame in [("train", train), ("validation", validation)]:
        normalized = frame.text.map(normalize_email)
        keys[name] = normalized
        audit["splits"][name] = {
            "rows": int(len(frame)),
            "ham": int((frame.label == 0).sum()), "spam": int((frame.label == 1).sum()),
            "raw_duplicate_rows": int(frame.text.duplicated().sum()),
            "normalized_duplicate_rows": int(normalized.duplicated().sum()),
            "normalized_conflicting_groups": int(
                (frame.assign(key=normalized).groupby("key").label.nunique() > 1).sum()),
            "sha256": sha256(data_dir / f"{name}.csv"),
        }
    raw_overlap = set(train.text) & set(validation.text)
    overlap = set(keys["train"]) & set(keys["validation"])
    keep = ~keys["validation"].isin(overlap).to_numpy()
    audit["raw_cross_split_unique_overlap"] = len(raw_overlap)
    audit["normalized_cross_split_unique_overlap"] = len(overlap)
    audit["overlapping_validation_rows"] = int((~keep).sum())
    audit["note"] = ("Normalized duplicates remain in Member 1's fixed splits. "
        "No rows were removed or reassigned. Reported primary metrics use all "
        "validation rows; overlap-excluded metrics are a sensitivity check only. "
        "Member 1/5 should review decoded-text grouping before final evaluation. "
        "No independent exhaustive near-duplicate or test-split audit was run here.")
    records = []
    for key in sorted(overlap):
        for split, frame in [("train", train), ("validation", validation)]:
            for index in frame.index[keys[split] == key]:
                records.append({"split": split, "row_index_0based": int(index),
                    "label": int(frame.loc[index, "label"]),
                    "normalized_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    "normalized_text_excerpt": key[:650]})
    pd.DataFrame(records, columns=["split", "row_index_0based", "label",
        "normalized_sha256", "normalized_text_excerpt"]).to_csv(
        out / "normalization_overlap.csv", index=False)
    write_json(out / "data_audit.json", audit)
    return keep


def make_figures(experiments, best, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    cm = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=17,
                color="white" if value > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Legitimate (0)", "Spam (1)"],
        yticklabels=["Legitimate (0)", "Spam (1)"],
        xlabel="Predicted label", ylabel="Actual label",
        title="Naive Bayes on the shared validation set")
    fig.savefig(out / "validation_confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), layout="constrained")
    for (ngram, min_df, sublinear), group in experiments.groupby(
            ["ngram", "min_df", "sublinear_tf"], sort=True):
        group = group.sort_values("alpha")
        ax.plot(group.alpha, 100 * group.f1, marker="o", markersize=4,
                label=f"{ngram}; min_df={min_df}; log TF={sublinear}")
    ax.set(xscale="log", xlabel="Smoothing alpha (log scale)",
        ylabel="Spam F1 (%)", title="Validation results for all 32 configurations")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2, loc="lower center")
    fig.savefig(out / "validation_tuning.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = load_split(args.data_dir / "train.csv")
    val = load_split(args.data_dir / "validation.csv")
    keep = audit_data(train, val, args.data_dir, out)
    print(f"Loaded {len(train)} training and {len(val)} validation emails.", flush=True)
    print(f"Normalized overlap: {(~keep).sum()} validation row(s); fixed splits retained.", flush=True)
    print("Test set is not read. Selection: highest spam F1, then precision, then lower experiment ID.", flush=True)
    rows = []
    best_key = None
    best_pipeline = None
    best_row = None
    # Fit each vectorizer once and reuse its sparse matrices across four alphas.
    for ngram, min_df, sublinear in itertools.product([(1, 1), (1, 2)], [2, 5], [False, True]):
        vectorizer = make_vectorizer(ngram, min_df, sublinear)
        t = time.perf_counter()
        x_train = vectorizer.fit_transform(train.text)
        vectorizer_fit_seconds = time.perf_counter() - t
        t = time.perf_counter()
        x_val = vectorizer.transform(val.text)
        validation_transform_seconds = time.perf_counter() - t
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            classifier = MultinomialNB(alpha=alpha, fit_prior=True)
            t = time.perf_counter()
            classifier.fit(x_train, train.label)
            classifier_fit_seconds = time.perf_counter() - t
            t = time.perf_counter()
            prediction = classifier.predict(x_val)
            prediction_seconds = time.perf_counter() - t
            probability = classifier.predict_proba(x_val)[:, 1]
            number = len(rows) + 1
            row = {"experiment_id": number, "ngram": f"{ngram[0]}-{ngram[1]}",
                "min_df": min_df, "sublinear_tf": sublinear, "alpha": alpha,
                "max_features": 50000, "vocabulary_size": len(vectorizer.vocabulary_),
                "vectorizer_fit_seconds": vectorizer_fit_seconds,
                "classifier_fit_seconds": classifier_fit_seconds,
                "total_fit_seconds": vectorizer_fit_seconds + classifier_fit_seconds,
                "validation_transform_seconds": validation_transform_seconds,
                "classifier_predict_seconds": prediction_seconds,
                **metrics(val.label, prediction, probability)}
            rows.append(row)
            key = (row["f1"], row["precision"], -number)
            if best_key is None or key > best_key:
                best_key, best_row = key, row.copy()
                best_pipeline = Pipeline([("tfidf", vectorizer), ("nb", classifier)])
            print(f"{number:02d}/32 ngram={ngram} min_df={min_df} log_tf={sublinear} "
                  f"alpha={alpha:g} F1={row['f1']:.6f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "experiments.csv", index=False)
    model_path = out / "naive_bayes_pipeline.joblib"
    joblib.dump(best_pipeline, model_path, compress=3)
    prediction = best_pipeline.predict(val.text)
    probability = best_pipeline.predict_proba(val.text)[:, 1]
    reloaded = joblib.load(model_path)
    if not np.array_equal(prediction, reloaded.predict(val.text)):
        raise AssertionError("Reload changed predictions")
    np.testing.assert_allclose(probability, reloaded.predict_proba(val.text)[:, 1], rtol=0, atol=0)
    vectorizer = best_pipeline.named_steps["tfidf"]
    config = {"ngram_range": list(vectorizer.ngram_range), "min_df": vectorizer.min_df,
        "sublinear_tf": vectorizer.sublinear_tf, "max_features": vectorizer.max_features}
    transformed = vectorizer.transform(val.text)
    validation_table = pd.DataFrame({"row_index_0based": val.index,
        "true_label": val.label, "predicted_label": prediction,
        "spam_probability": probability,
        "normalized_train_overlap": ~keep,
        "recognized_feature_count": transformed.getnnz(axis=1),
        "text_excerpt": val.text.map(normalize_email).str.slice(0, 900)})
    validation_table["error_type"] = np.select(
        [(val.label == 0) & (prediction == 1), (val.label == 1) & (prediction == 0)],
        ["false_positive", "false_negative"], default="correct")
    # The full email text already exists in validation.csv; retain excerpts only
    # for the much smaller error-analysis file. Row IDs allow exact reconstruction.
    validation_table.drop(columns="text_excerpt").to_csv(out / "validation_predictions.csv", index=False)
    errors = validation_table.loc[validation_table.error_type != "correct"].copy()
    errors.to_csv(out / "validation_errors.csv", index=False)
    classifier = best_pipeline.named_steps["nb"]
    ratio = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    feature_table = pd.DataFrame({"feature": vectorizer.get_feature_names_out(),
        "spam_vs_ham_log_probability_ratio": ratio})
    feature_table = pd.concat([feature_table.nlargest(25, "spam_vs_ham_log_probability_ratio"),
        feature_table.nsmallest(25, "spam_vs_ham_log_probability_ratio")])
    feature_table.to_csv(out / "indicative_features.csv", index=False)
    experiments = pd.DataFrame(rows)
    baseline = next(r for r in rows if r["ngram"] == "1-1" and r["min_df"] == 2
                    and not r["sublinear_tf"] and r["alpha"] == 1.0)
    majority = int(train.label.mode().iloc[0])
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "validation", "test_evaluated": False,
        "positive_label": 1, "label_mapping": {"0": "Legitimate", "1": "Spam"},
        "selection_rule": "highest validation spam F1; then precision; then lower experiment ID",
        "decision_rule": "MultinomialNB.predict argmax; ties resolve to class 0; no threshold tuning",
        "probability_note": "Model estimates, not calibrated probabilities",
        "number_of_configurations": len(rows), "baseline": baseline, "selected": best_row,
        "tfidf_config": config,
        "validation_without_normalized_train_overlap": metrics(
            val.label.to_numpy()[keep], prediction[keep], probability[keep]),
        "majority_baseline": {"predicted_label": majority,
            **metrics(val.label, np.full(len(val), majority))},
        "validation_zero_feature_rows": int((transformed.getnnz(axis=1) == 0).sum()),
        "save_reload_predictions_and_probabilities_identical": True,
        "fit_scope": "training only; no refit on combined train and validation",
        "total_run_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "joblib": joblib.__version__, "platform": platform.platform(),
            "processor": platform.processor()},
        "source_sha256": {p.name: sha256(p) for p in [ROOT/"preprocessing.py", ROOT/"train_model_1.py"]},
        "pipeline_sha256": sha256(model_path),
    }
    write_json(out / "validation_results.json", results)
    make_figures(experiments, best_row, out)
    print("SELECTED", json.dumps(best_row), flush=True)
    print(f"Saved model and validation evidence to {out}", flush=True)


if __name__ == "__main__":
    main()
