# Spam Detection: Neural Network Model (Handover)

Binary spam classifier (**0 = ham, 1 = spam**): a Keras multilayer perceptron on TF-IDF features, using the Enron spam partitions prepared by the data member.

## 1. Folder contents

```
project/
├── train_model_3.py             # the whole pipeline (three stages)
├── requirements.txt
├── README.md
├── nn_report_section_skeleton.md  # report outline, to be filled with real results
├── data/                        # YOU PROVIDE: train.csv, validation.csv, test.csv
└── nn_artifacts/                # created by the script
```

Use the **original, unmodified** CSVs (columns `text`, `label`). Never re-split, shuffle or re-clean them: all models are compared on identical partitions.

## 2. Setup

Python 3.10 to 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 3. Run the three stages in order

| Stage | Command | Uses | Does |
|---|---|---|---|
| 1 | `python train_model_3.py tune` | train + validation | 16-trial grid (learning rate, units, dropout, batch size); picks the lowest validation loss; writes `experiments/tuning_log.csv` and `best_config.json`. Resumes if interrupted. |
| 2 | `python train_model_3.py seeds` | train + validation | Repeats the chosen config with seeds 42 to 46 (change with `--seeds 1 2 3`); writes `seed_runs.csv`, `seed_summary.json`, seed learning curves. |
| 3 | `python train_model_3.py final` | train + validation + **test** | Trains the chosen config (seed 42), picks the threshold on validation, scores test **once**, saves model, vectorizer, config, predictions, errors, plots, `results.json`, `report_tables.md`. |

`test.csv` is not even loaded in stages 1 and 2. After stage 3 a marker file (`experiments/FINAL_TEST_DONE.txt`) blocks further runs of `final`. Do not use `--force` to retune after seeing test results.

Runtime: stage 1 trains 16 networks (the longest stage); stages 2 and 3 are shorter. Training time per run is recorded in the logs.

## 4. Rules that must not be broken

- The TF-IDF vocabulary and IDF weights are fit on `train.csv` only.
- Validation is used for all selection: tuning, early stopping, decision threshold.
- Test is scored once, for the frozen configuration.
- Do not use the earlier full-data TF-IDF matrix as classifier features (it was for similarity auditing only).

## 5. Reproducibility

- Seed 42 for tuning and the final model; seeds 42 to 46 for the stability check. The handover specified no seed, so 42 is the project choice.
- Deterministic TensorFlow ops are on; results may still differ slightly across hardware or library versions. `results.json` records versions.

## 6. Outputs in `nn_artifacts/`

| File | Use |
|---|---|
| `report_tables.md` | Markdown tables (tuning, seeds, final results) generated from the real runs, ready for the report |
| `results.json` | Final summary: config, training time, validation and test metrics, versions |
| `experiments/tuning_log.csv`, `best_config.json` | Every tuning trial and the selected configuration |
| `experiments/seed_runs.csv`, `seed_summary.json`, `seed_histories/` | Per-seed results, mean and std, per-seed curves |
| `plots/training_curves.png`, `seed_learning_curves.png`, `tuning_trials.png` | Learning curves and tuning overview |
| `plots/confusion_matrix_*.png`, `roc_pr_curves.png`, `test_score_distribution.png`, `threshold_sweep_validation.png` | Evaluation plots |
| `predictions/*_predictions.csv`, `*_misclassified.csv` | Every prediction, and errors sorted by confidence |
| `classification_report.txt`, `model_summary.txt`, `tfidf_vocabulary.csv` | Per-class report, architecture, vocabulary with IDF |
| `model/spam_nn.keras`, `tfidf_vectorizer.joblib`, `config.json` | Saved model, text processor, configuration |

## 7. Using the saved model

```python
import json, joblib, tensorflow as tf

vec = joblib.load("nn_artifacts/model/tfidf_vectorizer.joblib")
model = tf.keras.models.load_model("nn_artifacts/model/spam_nn.keras")
threshold = json.load(open("nn_artifacts/model/config.json"))["threshold"]

emails = ["Subject: you have won a prize, click here now"]
prob = model.predict(vec.transform(emails).toarray(), verbose=0).ravel()
print(prob, (prob >= threshold).astype(int))   # 1 = spam
```

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Missing data/train.csv ...` | Put the three CSVs in `data/` next to the script. |
| `No best_config.json found` | Run stage 1 (`tune`) first. |
| `The test set was already scored` | Intended. Only use `--force` after a crashed run. |
| `No matching distribution found for tensorflow` | Use Python 3.10 to 3.12. |
| Out-of-memory errors | Lower `MAX_FEATURES` or the batch sizes in `GRID` at the top of the script. |
| Stage 1 too slow | Shrink `GRID` (fewer values per setting). Training on GPU is slower with deterministic ops; comment out `enable_op_determinism()` if speed matters more. |

## 9. Data notes

About two thirds of the `text` values are stringified Python lists (for example `['Subject: ...\n']`); the script leaves this as is. No duplicates within or across partitions, and no missing values.
