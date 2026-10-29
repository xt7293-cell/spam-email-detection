"""
Neural network (Keras MLP on TF-IDF) for spam detection, 0 = ham, 1 = spam.

Folder layout:
    train_model_3.py
    data/            train.csv  validation.csv  test.csv      (input, read-only)
    nn_artifacts/                                             (output, created)
        experiments/   tuning_log.csv, best_config.json, seed_runs.csv, ...
        plots/  predictions/  model/  results.json  report_tables.md ...

Three stages, run in this order:

    python train_model_3.py tune     # small grid search, VALIDATION only
    python train_model_3.py seeds    # repeat the chosen config with several seeds, VALIDATION only
    python train_model_3.py final    # train the chosen config once, score TEST once, save everything

Handover rules followed:
  * same train / validation / test partitions as every other model (never re-split)
  * every trainable transformation (TF-IDF vocabulary + IDF weights) is fit on TRAIN ONLY
  * validation is used for all model selection (tuning, early stopping, threshold)
  * test.csv is NOT loaded during `tune` or `seeds`; it is scored exactly once, in `final`
"""
import argparse
import itertools
import json
import platform
import random
import sys
import time
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
SEED = 1                                  # primary seed
SEEDS = [1, 2, 3, 4, 5]               # seeds for the stability check
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parents[1] / "data" / "splits"
ARTIFACTS_DIR = ROOT / "nn_artifacts"
EXP_DIR = ARTIFACTS_DIR / "experiments"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PRED_DIR = ARTIFACTS_DIR / "predictions"
MODEL_DIR = ARTIFACTS_DIR / "model"

MAX_FEATURES = 20000                       # fixed input representation (not tuned)
MAX_EPOCHS = 30
PATIENCE = 3

# Limited tuning grid: learning rate, hidden units, dropout, batch size (2 x 2 x 2 x 2 = 16 trials)
GRID = {
    "learning_rate": [1e-3, 3e-4],
    "units": [64, 256],                    # first hidden layer; second = units // 4 (min 16)
    "dropout": [0.3, 0.5],                 # applied after both hidden layers
    "batch_size": [64, 128],
}

tf.config.experimental.enable_op_determinism()


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --- TF-FREE HELPERS -----------------------------------------------------------
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


def choose_threshold(y_val, val_prob):
    """Decision threshold = the one with the best VALIDATION F1."""
    sweep = threshold_sweep(y_val, val_prob, np.round(np.arange(0.05, 0.96, 0.05), 2))
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"]), sweep


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
    err["confidence"] = np.where(err["pred"] == 1, err["prob_spam"], 1 - err["prob_spam"])
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


def plot_seed_curves(histories, path):
    """histories: {seed: hist_df}. Overlay train/validation loss for every seed."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for seed, h in histories.items():
        ax[0].plot(h["epoch"], h["loss"], marker=".", label=f"seed {seed}")
        ax[1].plot(h["epoch"], h["val_loss"], marker=".", label=f"seed {seed}")
    ax[0].set_title("Training loss"); ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tuning(tuning_df, path):
    d = tuning_df.sort_values("best_val_loss").reset_index(drop=True)
    labels = [f"lr={r.learning_rate:g}, u={r.units}, d={r.dropout:g}, b={r.batch_size}"
              for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
    ax.barh(range(len(d)), d["best_val_loss"])
    ax.set_yticks(range(len(d))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("best validation loss (lower is better)")
    ax.set_title("Tuning trials (validation only)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, prob, t, title, path):
    pred = (prob >= t).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, norm, sub in [(ax[0], None, "counts"), (ax[1], "true", "row-normalised")]:
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred, display_labels=["ham", "spam"], normalize=norm,
            cmap="Blues", ax=a, values_format="d" if norm is None else ".3f")
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


def md_table(df, floatfmt=".4f"):
    """Minimal markdown table (avoids the optional 'tabulate' dependency)."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summarise_seed_runs(seed_df):
    """Mean and standard deviation of each numeric column across seeds."""
    cols = ["best_val_loss", "accuracy", "precision", "recall", "f1", "roc_auc",
            "false_positive_rate", "false_negative_rate", "best_epoch",
            "epochs_run", "train_seconds"]
    return {c: {"mean": float(seed_df[c].mean()), "std": float(seed_df[c].std(ddof=1))}
            for c in cols}


def make_report_tables(tuning_df, seed_df, seed_summary, results):
    """Markdown tables filled with REAL numbers from the runs, for the report."""
    parts = []
    if tuning_df is not None:
        t = tuning_df.sort_values("best_val_loss")[
            ["learning_rate", "units", "dropout", "batch_size", "best_val_loss",
             "f1", "roc_auc", "best_epoch", "train_seconds"]]
        parts += ["## Tuning trials (validation only, sorted by validation loss)", md_table(t), ""]
    if seed_df is not None:
        s = seed_df[["seed", "best_val_loss", "accuracy", "precision", "recall", "f1",
                     "roc_auc", "best_epoch", "train_seconds"]]
        parts += ["## Stability across seeds (validation)", md_table(s), ""]
        rows = [{"metric": k, "mean": v["mean"], "std": v["std"]}
                for k, v in seed_summary.items()]
        parts += ["### Mean and standard deviation", md_table(pd.DataFrame(rows)), ""]
    if results is not None:
        rows = [{"metric": k, "validation": results["validation"][k], "test": results["test"][k]}
                for k in ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
                          "pr_auc", "false_positive_rate", "false_negative_rate",
                          "tn", "fp", "fn", "tp"]]
        parts += ["## Final model (single run, seed %d)" % results["seed"],
                  md_table(pd.DataFrame(rows)), "",
                  "Training time: %.1f s over %d epochs (best epoch %d)."
                  % (results["training"]["train_seconds"], results["training"]["epochs_run"],
                     results["training"]["best_epoch"]), ""]
    return "\n".join(parts)
# --- END TF-FREE HELPERS --------------------------------------------------------


# ----------------------------------------------------------------------------
# Data and features
# ----------------------------------------------------------------------------
def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        sys.exit(f"Missing {path} - put train.csv, validation.csv and test.csv in '{DATA_DIR}/'.")
    df = pd.read_csv(path)
    assert list(df.columns) == ["text", "label"], f"{name}: unexpected columns"
    assert set(df["label"].unique()) <= {0, 1}, f"{name}: labels must be 0/1"
    print(f"{name:10s} rows={len(df):6d}  spam share={df['label'].mean():.3f}")
    return df


def build_features(train, val, test=None):
    """TF-IDF fitted on TRAIN ONLY; validation (and test, if given) only transformed."""
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2, strip_accents="unicode")
    X_train = vec.fit_transform(train["text"])        # fit ONLY here
    X_val = vec.transform(val["text"])
    X_test = vec.transform(test["text"]) if test is not None else None
    print("TF-IDF matrix (train):", X_train.shape)
    return vec, X_train, X_val, X_test


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


# ----------------------------------------------------------------------------
# Model + one training run
# ----------------------------------------------------------------------------
def build_model(n_features, cfg):
    h2 = max(cfg["units"] // 4, 16)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(cfg["units"], activation="relu"),
        tf.keras.layers.Dropout(cfg["dropout"]),
        tf.keras.layers.Dense(h2, activation="relu"),
        tf.keras.layers.Dropout(cfg["dropout"]),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def train_one(cfg, seed, X_train, y_train, X_val, y_val, verbose=0):
    """Train one configuration with one seed. Uses TRAIN for fitting, VALIDATION for early stopping."""
    tf.keras.backend.clear_session()
    set_all_seeds(seed)
    model = build_model(X_train.shape[1], cfg)
    train_gen = SparseBatches(X_train, y_train, cfg["batch_size"], shuffle=True, seed=seed)
    val_gen = SparseBatches(X_val, y_val, cfg["batch_size"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    t0 = time.perf_counter()
    history = model.fit(train_gen, validation_data=val_gen, epochs=MAX_EPOCHS,
                        callbacks=[early_stop], verbose=verbose)
    train_seconds = time.perf_counter() - t0

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    best_epoch = int(hist_df.loc[hist_df["val_loss"].idxmin(), "epoch"])
    val_prob = model.predict(SparseBatches(X_val, y_val, 512), verbose=0).ravel()
    info = {"train_seconds": round(train_seconds, 2), "epochs_run": len(hist_df),
            "best_epoch": best_epoch, "best_val_loss": float(hist_df["val_loss"].min())}
    return model, hist_df, val_prob, info


def env_info():
    return {"python": platform.python_version(), "tensorflow": tf.__version__,
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}


def load_best_config():
    path = EXP_DIR / "best_config.json"
    if not path.exists():
        sys.exit("No best_config.json found. Run `python train_model_3.py tune` first.")
    return json.load(open(path))["config"]


# ----------------------------------------------------------------------------
# Stage 1: tune (validation only; test.csv is never loaded)
# ----------------------------------------------------------------------------
def run_tune():
    train, val = load_split("train"), load_split("validation")
    y_train = train["label"].to_numpy("float32")
    y_val = val["label"].to_numpy("float32")
    _, X_train, X_val, _ = build_features(train, val)

    log_path = EXP_DIR / "tuning_log.csv"
    done = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()   # resume support
    rows = done.to_dict("records")
    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))

    for i, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        if any(all(r[k] == cfg[k] for k in keys) for r in rows):
            print(f"[{i}/{len(combos)}] {cfg} already done, skipping")
            continue
        print(f"[{i}/{len(combos)}] {cfg}")
        _, _, val_prob, info = train_one(cfg, SEED, X_train, y_train, X_val, y_val)
        t, _ = choose_threshold(y_val, val_prob)
        m = metrics_at(y_val, val_prob, t)
        rows.append({"trial": i, **cfg, "seed": SEED, **info, **m})
        pd.DataFrame(rows).to_csv(log_path, index=False)       # saved after every trial
        print(f"    val_loss={info['best_val_loss']:.4f}  f1={m['f1']:.4f}  "
              f"epochs={info['epochs_run']}  time={info['train_seconds']}s")

    df = pd.DataFrame(rows)
    # Selection rule: lowest validation loss (the same quantity early stopping monitors); ties -> higher F1
    best = df.sort_values(["best_val_loss", "f1"], ascending=[True, False]).iloc[0]
    best_cfg = {"learning_rate": float(best["learning_rate"]), "units": int(best["units"]),
                "dropout": float(best["dropout"]), "batch_size": int(best["batch_size"])}
    with open(EXP_DIR / "best_config.json", "w") as f:
        json.dump({"config": best_cfg,
                   "selection_rule": "lowest validation loss over the grid; ties broken by validation F1",
                   "best_val_loss": float(best["best_val_loss"]),
                   "val_f1": float(best["f1"]), "trials_run": len(df),
                   "grid": GRID, "tuning_seed": SEED}, f, indent=2)
    plot_tuning(df, PLOTS_DIR / "tuning_trials.png")
    print("\nBest configuration:", best_cfg)
    print("Next: python train_model_3.py seeds")


# ----------------------------------------------------------------------------
# Stage 2: stability across seeds (validation only)
# ----------------------------------------------------------------------------
def run_seeds(seeds):
    cfg = load_best_config()
    train, val = load_split("train"), load_split("validation")
    y_train = train["label"].to_numpy("float32")
    y_val = val["label"].to_numpy("float32")
    _, X_train, X_val, _ = build_features(train, val)

    (EXP_DIR / "seed_histories").mkdir(exist_ok=True)
    rows, histories = [], {}
    for seed in seeds:
        print(f"Seed {seed} with config {cfg}")
        _, hist_df, val_prob, info = train_one(cfg, seed, X_train, y_train, X_val, y_val)
        t, _ = choose_threshold(y_val, val_prob)
        m = metrics_at(y_val, val_prob, t)
        rows.append({"seed": seed, **info, **m})
        histories[seed] = hist_df
        hist_df.to_csv(EXP_DIR / "seed_histories" / f"history_seed_{seed}.csv", index=False)
        print(f"    val_loss={info['best_val_loss']:.4f}  f1={m['f1']:.4f}  time={info['train_seconds']}s")

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(EXP_DIR / "seed_runs.csv", index=False)
    summary = summarise_seed_runs(seed_df)
    with open(EXP_DIR / "seed_summary.json", "w") as f:
        json.dump({"config": cfg, "seeds": list(seeds), "summary": summary}, f, indent=2)
    plot_seed_curves(histories, PLOTS_DIR / "seed_learning_curves.png")
    print("\nF1 across seeds: mean %.4f, std %.4f" % (summary["f1"]["mean"], summary["f1"]["std"]))
    print("Next: python train_model_3.py final")


# ----------------------------------------------------------------------------
# Stage 3: final model + the one-time test evaluation
# ----------------------------------------------------------------------------
def run_final(force):
    marker = EXP_DIR / "FINAL_TEST_DONE.txt"
    if marker.exists() and not force:
        sys.exit(f"The test set was already scored ({marker.read_text().strip()}).\n"
                 "Scoring it again after seeing results is tuning on test. "
                 "Use --force only if you are certain (e.g. a crashed run).")
    cfg = load_best_config()
    if not (EXP_DIR / "seed_summary.json").exists():
        print("WARNING: no seed stability results found; run the `seeds` stage first.")

    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    y_train = train["label"].to_numpy("float32")
    y_val = val["label"].to_numpy("float32")
    y_test = test["label"].to_numpy("float32")
    vec, X_train, X_val, X_test = build_features(train, val, test)
    pd.DataFrame({"term": vec.get_feature_names_out(), "idf": vec.idf_}
                 ).sort_values("idf").to_csv(ARTIFACTS_DIR / "tfidf_vocabulary.csv", index=False)

    print(f"Final model: config {cfg}, seed {SEED}")
    model, hist_df, val_prob, info = train_one(cfg, SEED, X_train, y_train, X_val, y_val, verbose=2)
    model.summary()
    with open(ARTIFACTS_DIR / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))
    hist_df.to_csv(ARTIFACTS_DIR / "training_history.csv", index=False)
    plot_training_history(hist_df, info["best_epoch"], PLOTS_DIR / "training_curves.png")

    # Threshold chosen on VALIDATION
    best_t, sweep = choose_threshold(y_val, val_prob)
    sweep.to_csv(ARTIFACTS_DIR / "threshold_sweep_validation.csv", index=False)
    plot_threshold_sweep(sweep, best_t, PLOTS_DIR / "threshold_sweep_validation.png")
    val_metrics = metrics_at(y_val, val_prob, best_t)

    # ---- the single, one-time TEST evaluation ----
    test_prob = model.predict(SparseBatches(X_test, y_test, 512), verbose=0).ravel()
    test_metrics = metrics_at(y_test, test_prob, best_t)
    marker.write_text(datetime.now().isoformat(timespec="seconds"))
    print("VALIDATION:", json.dumps(val_metrics, indent=2))
    print("TEST:", json.dumps(test_metrics, indent=2))

    save_predictions(val, val_prob, best_t, "validation", PRED_DIR)
    save_predictions(test, test_prob, best_t, "test", PRED_DIR)
    plot_confusion(y_val, val_prob, best_t, "Validation", PLOTS_DIR / "confusion_matrix_validation.png")
    plot_confusion(y_test, test_prob, best_t, "Test", PLOTS_DIR / "confusion_matrix_test.png")
    plot_roc_pr({"validation": (y_val, val_prob), "test": (y_test, test_prob)},
                PLOTS_DIR / "roc_pr_curves.png")
    plot_score_distribution(y_test, test_prob, best_t, PLOTS_DIR / "test_score_distribution.png")

    with open(ARTIFACTS_DIR / "classification_report.txt", "w") as f:
        for name, y, p in [("VALIDATION", y_val, val_prob), ("TEST", y_test, test_prob)]:
            f.write(f"=== {name} (threshold {best_t:.2f}) ===\n")
            f.write(classification_report(y, (p >= best_t).astype(int),
                                          target_names=["ham", "spam"], digits=4) + "\n")

    model.save(MODEL_DIR / "spam_nn.keras")
    joblib.dump(vec, MODEL_DIR / "tfidf_vectorizer.joblib")
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump({"hyperparameters": cfg, "tfidf": {"max_features": MAX_FEATURES,
                   "ngram_range": [1, 2], "sublinear_tf": True, "min_df": 2,
                   "strip_accents": "unicode"}, "threshold": best_t, "seed": SEED}, f, indent=2)

    results = {
        "model": "MLP on TF-IDF (Keras)",
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "data": {n: {"rows": len(d), "spam_share": round(float(d["label"].mean()), 4)}
                 for n, d in [("train", train), ("validation", val), ("test", test)]},
        "config": {**cfg, "max_features": MAX_FEATURES, "ngram_range": [1, 2],
                   "sublinear_tf": True, "min_df": 2, "max_epochs": MAX_EPOCHS,
                   "early_stopping_patience": PATIENCE, "optimizer": "Adam",
                   "hidden_layers": [cfg["units"], max(cfg["units"] // 4, 16)],
                   "trainable_params": int(model.count_params())},
        "training": {"epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
                     "train_seconds": info["train_seconds"]},
        "threshold_selected_on_validation": best_t,
        "validation": val_metrics,
        "test": test_metrics,
        "environment": env_info(),
    }
    with open(ARTIFACTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    tuning_df = pd.read_csv(EXP_DIR / "tuning_log.csv") if (EXP_DIR / "tuning_log.csv").exists() else None
    seed_df = pd.read_csv(EXP_DIR / "seed_runs.csv") if (EXP_DIR / "seed_runs.csv").exists() else None
    seed_summary = (json.load(open(EXP_DIR / "seed_summary.json"))["summary"]
                    if seed_df is not None else None)
    (ARTIFACTS_DIR / "report_tables.md").write_text(
        make_report_tables(tuning_df, seed_df, seed_summary, results))
    print(f"\nAll artifacts written to '{ARTIFACTS_DIR}/'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "seeds", "final"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help="seeds for the stability check (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="allow `final` to run again after the test set was already scored")
    args = ap.parse_args()
    for d in (ARTIFACTS_DIR, EXP_DIR, PLOTS_DIR, PRED_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    {"tune": run_tune, "seeds": lambda: run_seeds(args.seeds),
     "final": lambda: run_final(args.force)}[args.mode]()


if __name__ == "__main__":
    main()
