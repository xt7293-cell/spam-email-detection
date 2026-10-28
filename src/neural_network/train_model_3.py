"""
Neural network (Keras MLP on TF-IDF) for spam detection, 0 = ham, 1 = spam.

Folder layout expected:
    spam_nn.py
    data/        train.csv  validation.csv  test.csv     (input, read-only)
    nn_artifacts/   ...everything this script produces...  (output, created)

Handover rules followed:
  * same train / validation / test partitions as every other model (never re-split)
  * every trainable transformation (TF-IDF vocabulary + IDF weights) is fit on TRAIN ONLY
  * validation is used for model selection (early stopping, decision threshold)
  * test is used exactly once, at the very end, for the final unbiased evaluation
"""
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             RocCurveDisplay, PrecisionRecallDisplay)

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------
SEED = 42                       # project seed (none was specified in the handover)
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("nn_artifacts")
MAX_FEATURES = 20000
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 3

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

PLOTS_DIR = ARTIFACTS_DIR / "plots"
PRED_DIR = ARTIFACTS_DIR / "predictions"
MODEL_DIR = ARTIFACTS_DIR / "model"
for d in (ARTIFACTS_DIR, PLOTS_DIR, PRED_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --- ARTIFACT HELPERS (no TensorFlow dependency) -----------------------------
def metrics_at(y_true, prob, t):
    """Scalar + confusion-matrix metrics at decision threshold t."""
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "false_positive_rate": float(fp / (fp + tn)),   # ham flagged as spam
        "false_negative_rate": float(fn / (fn + tp)),   # spam missed
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_sweep(y_true, prob, thresholds):
    return pd.DataFrame([metrics_at(y_true, prob, t) for t in thresholds])


def save_predictions(df, prob, t, split, out_dir):
    """Per-email predictions plus a file with only the mistakes."""
    pred = (prob >= t).astype(int)
    out = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "label": df["label"].to_numpy(),
        "prob_spam": prob,
        "pred": pred,
        "correct": pred == df["label"].to_numpy(),
        "text_preview": df["text"].str.slice(0, 300).to_numpy(),
    })
    out.to_csv(out_dir / f"{split}_predictions.csv", index=False)

    err = out[~out["correct"]].copy()
    err["error_type"] = np.where(err["label"] == 0,
                                 "false_positive (ham->spam)",
                                 "false_negative (spam->ham)")
    err["confidence"] = np.where(err["pred"] == 1, err["prob_spam"],
                                 1 - err["prob_spam"])
    err.sort_values("confidence", ascending=False).to_csv(
        out_dir / f"{split}_misclassified.csv", index=False)


def plot_training_history(hist_df, best_epoch, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in [(ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")]:
        a.plot(hist_df["epoch"], hist_df[key], marker="o", label="train")
        a.plot(hist_df["epoch"], hist_df[f"val_{key}"], marker="o", label="validation")
        a.axvline(best_epoch, ls="--", c="grey", label=f"best epoch ({best_epoch})")
        a.set_xlabel("epoch"); a.set_title(title); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, prob, t, title, path):
    pred = (prob >= t).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, norm, sub in [(ax[0], None, "counts"), (ax[1], "true", "row-normalised")]:
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred, display_labels=["ham", "spam"],
            normalize=norm, cmap="Blues", ax=a,
            values_format="d" if norm is None else ".3f")
        a.set_title(f"{title} ({sub})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc_pr(curves, path):
    """curves: {name: (y_true, prob)}"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, p) in curves.items():
        RocCurveDisplay.from_predictions(y, p, name=name, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y, p, name=name, ax=ax[1])
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_title("ROC curve"); ax[1].set_title("Precision-recall curve")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_score_distribution(y_true, prob, t, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob[y_true == 0], bins=40, alpha=0.6, label="ham (0)")
    ax.hist(prob[y_true == 1], bins=40, alpha=0.6, label="spam (1)")
    ax.axvline(t, c="k", ls="--", label=f"threshold {t:.2f}")
    ax.set_yscale("log"); ax.set_xlabel("predicted P(spam)")
    ax.set_ylabel("emails (log scale)"); ax.set_title("Test score distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_threshold_sweep(sweep_df, best_t, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ["precision", "recall", "f1", "accuracy"]:
        ax.plot(sweep_df["threshold"], sweep_df[col], marker=".", label=col)
    ax.axvline(best_t, c="k", ls="--", label=f"chosen ({best_t:.2f})")
    ax.set_xlabel("decision threshold"); ax.set_title("Validation: threshold sweep")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
# --- END ARTIFACT HELPERS -----------------------------------------------------


# ----------------------------------------------------------------------------
# 1. Load the fixed partitions (never re-split)
# ----------------------------------------------------------------------------
for name in ("train", "validation", "test"):
    if not (DATA_DIR / f"{name}.csv").exists():
        sys.exit(f"Missing {DATA_DIR / (name + '.csv')} - put the three CSVs in '{DATA_DIR}/'.")

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

split_summary = {}
for name, df in [("train", train), ("validation", val), ("test", test)]:
    assert list(df.columns) == ["text", "label"], f"{name}: unexpected columns"
    assert set(df["label"].unique()) <= {0, 1}, f"{name}: labels must be 0/1"
    split_summary[name] = {"rows": len(df), "spam_share": round(float(df["label"].mean()), 4)}
    print(f"{name:10s} rows={len(df):6d}  spam share={df['label'].mean():.3f}")

y_train = train["label"].to_numpy(dtype="float32")
y_val = val["label"].to_numpy(dtype="float32")
y_test = test["label"].to_numpy(dtype="float32")

# ----------------------------------------------------------------------------
# 2. Features: fit on TRAIN only, transform val/test
# ----------------------------------------------------------------------------
model_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
)
X_train = model_vectorizer.fit_transform(train["text"])   # fit ONLY here
X_val = model_vectorizer.transform(val["text"])
X_test = model_vectorizer.transform(test["text"])
print("TF-IDF matrix (train):", X_train.shape)

pd.DataFrame({"term": model_vectorizer.get_feature_names_out(),
              "idf": model_vectorizer.idf_}
             ).sort_values("idf").to_csv(ARTIFACTS_DIR / "tfidf_vocabulary.csv", index=False)


class SparseBatches(tf.keras.utils.Sequence):
    """Feeds a scipy sparse matrix to Keras in dense mini-batches."""

    def __init__(self, X, y, batch_size, shuffle=False, seed=SEED):
        super().__init__()
        self.X, self.y, self.bs, self.shuffle = X, y, batch_size, shuffle
        self.rng = np.random.RandomState(seed)
        self.idx = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.bs))

    def __getitem__(self, i):
        sel = self.idx[i * self.bs:(i + 1) * self.bs]
        return self.X[sel].toarray().astype("float32"), self.y[sel]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.idx)


train_gen = SparseBatches(X_train, y_train, BATCH_SIZE, shuffle=True)
val_gen = SparseBatches(X_val, y_val, BATCH_SIZE)

# ----------------------------------------------------------------------------
# 3. Model
# ----------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()
with open(ARTIFACTS_DIR / "model_summary.txt", "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# ----------------------------------------------------------------------------
# 4. Train. Validation drives early stopping / model selection.
# ----------------------------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)
history = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS,
    callbacks=[early_stop], verbose=2,
)

hist_df = pd.DataFrame(history.history)
hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
hist_df.to_csv(ARTIFACTS_DIR / "training_history.csv", index=False)
best_epoch = int(hist_df.loc[hist_df["val_loss"].idxmin(), "epoch"])
plot_training_history(hist_df, best_epoch, PLOTS_DIR / "training_curves.png")
print(f"Stopped after {len(hist_df)} epochs; best epoch = {best_epoch}")

# ----------------------------------------------------------------------------
# 5. Model selection on VALIDATION (decision threshold)
# ----------------------------------------------------------------------------
val_prob = model.predict(val_gen, verbose=0).ravel()
sweep = threshold_sweep(y_val, val_prob, np.round(np.arange(0.05, 0.96, 0.05), 2))
sweep.to_csv(ARTIFACTS_DIR / "threshold_sweep_validation.csv", index=False)
best_t = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
plot_threshold_sweep(sweep, best_t, PLOTS_DIR / "threshold_sweep_validation.png")
print(f"Chosen threshold (max validation F1): {best_t:.2f}")

val_metrics = metrics_at(y_val, val_prob, best_t)
print("VALIDATION:", json.dumps(val_metrics, indent=2))

# ----------------------------------------------------------------------------
# 6. FINAL, one-time evaluation on TEST (do not tune after seeing this)
# ----------------------------------------------------------------------------
test_prob = model.predict(SparseBatches(X_test, y_test, BATCH_SIZE), verbose=0).ravel()
test_metrics = metrics_at(y_test, test_prob, best_t)
print("TEST:", json.dumps(test_metrics, indent=2))

# ----------------------------------------------------------------------------
# 7. Artifacts for evaluation & results analysis
# ----------------------------------------------------------------------------
# Predictions and error lists
save_predictions(val, val_prob, best_t, "validation", PRED_DIR)
save_predictions(test, test_prob, best_t, "test", PRED_DIR)

# Plots
plot_confusion(y_val, val_prob, best_t, "Validation", PLOTS_DIR / "confusion_matrix_validation.png")
plot_confusion(y_test, test_prob, best_t, "Test", PLOTS_DIR / "confusion_matrix_test.png")
plot_roc_pr({"validation": (y_val, val_prob), "test": (y_test, test_prob)},
            PLOTS_DIR / "roc_pr_curves.png")
plot_score_distribution(y_test, test_prob, best_t, PLOTS_DIR / "test_score_distribution.png")

# Per-class reports
with open(ARTIFACTS_DIR / "classification_report.txt", "w") as f:
    for name, y, p in [("VALIDATION", y_val, val_prob), ("TEST", y_test, test_prob)]:
        f.write(f"=== {name} (threshold {best_t:.2f}) ===\n")
        f.write(classification_report(y, (p >= best_t).astype(int),
                                      target_names=["ham", "spam"], digits=4))
        f.write("\n")

# Model + vectorizer (needed to reproduce predictions)
model.save(MODEL_DIR / "spam_nn.keras")
joblib.dump(model_vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

# Single results file the team can load to compare models
results = {
    "model": "MLP on TF-IDF (Keras)",
    "created": datetime.now().isoformat(timespec="seconds"),
    "seed": SEED,
    "data": split_summary,
    "config": {
        "max_features": MAX_FEATURES, "ngram_range": [1, 2], "sublinear_tf": True,
        "min_df": 2, "batch_size": BATCH_SIZE, "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE, "optimizer": "Adam(1e-3)",
        "architecture": "Dense256-Drop0.5-Dense64-Drop0.3-Sigmoid",
        "trainable_params": int(model.count_params()),
    },
    "training": {"epochs_run": len(hist_df), "best_epoch": best_epoch},
    "threshold_selected_on_validation": best_t,
    "validation": val_metrics,
    "test": test_metrics,
    "environment": {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "scikit_learn": sklearn.__version__, "numpy": np.__version__,
        "pandas": pd.__version__,
    },
}
with open(ARTIFACTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll artifacts written to '{ARTIFACTS_DIR}/'")