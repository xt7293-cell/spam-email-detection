# Spam Email Detection

Two-page email comparison website connected to the supplied saved models. Labels: 0 = legitimate, 1 = spam. Nothing trains when you start the website.

## Start with the Python launcher (Windows)

Install Python first, then run these commands in PowerShell:

```powershell
cd D:\GitHub\spam-email-detection
python start.py
```

The launcher creates `.venv` if needed, installs missing dependencies for all three models, checks for the saved model files, and starts the local server. Use the same command for first-time setup and later launches. The four saved runtime models are included in Git, so a fresh clone includes everything needed for inference once these files have been committed and pushed. The launcher installs dependencies; it does not retrain models.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep the terminal open and press **Ctrl+C** to stop.

```powershell
# Verify setup and model loading without starting the server
python start.py --check

# Use a different port
python start.py --port 8001
```

## Important compatibility limits

The submitted artifacts report scikit-learn 1.9.1, NumPy 2.5.4, and TensorFlow 2.22.0-rc0. The original Naive Bayes requirements and neural results.json are preserved. The root requirements are broad installation guidance, not a verified lockfile. Prefer matching the original training environment. Cross-version pickle loading may fail or produce warnings; check predictions in the original environment before reporting results. Only load model files you trust.

## Model status

- Naive Bayes: saved pipeline and custom `preprocessing.py` retained together under `src/naive_bayes`.
- Logistic regression: supplied pipeline, preprocessing, training code, tests and validation evidence in `src/logistic_regression/`. Loaded directly by the app; no test scores are claimed.
- Neural network: original Keras model and TF-IDF under `src/neural_network/nn_artifacts/model`. The decision threshold is loaded from the supplied results.json.

Each model retains its own fitted text processor. The Naive Bayes and neural network use different TF-IDF configurations; these supplied experiments are not a controlled shared-vectorizer comparison. Do not refit a vectorizer during inference. Subject and body are joined with a newline; use complete raw email text with an empty subject if matching the dataset format requires it.

## Layout

- `frontend/`: website input and comparison views.
- `app/`: local HTTP server and model adapters.
- `src/naive_bayes/`: original contribution, inference code, training code, tests, results and saved model.
- `src/neural_network/`: original neural contribution, code, results and saved artifacts.
- `src/logistic_regression/`: supplied logistic regression contribution and saved artifacts.
- Previous `data/processed/` splits have been removed; use the newly imported contribution datasets.
- `data/preparation_submission/`: separate preparation submission and its different CSV splits.
- `docs/contributions/`: original contribution documents.
- `docs/data_inventory.json`: source hashes, row counts and class counts.
- `docs/archive/`: earlier repository README preserved for reference.

## Data mismatch

The preparation-submission CSVs differ from the model-package CSVs. Preserve both; do not substitute one for the other or claim all experiments used the preparation-submission split. The inventory records both. The supplied neural package already contains test results; this integration does not rerun test evaluation or tune models.

To rerun Naive Bayes deliberately, use:

```
python src/naive_bayes/train_model_1.py --data-dir data/splits --output-dir src/naive_bayes/artifacts
```

The neural training script is preserved, including its original relative paths. Adapt DATA_DIR and ARTIFACTS_DIR before deliberately retraining; do not import the training script from the web application because its training runs at module scope.

## Clone or pull and run

Saved runtime models are included through explicit exceptions in `.gitignore`. Experimental models, training data, and the local `.venv` remain excluded. Commit and push the four model files with the repository changes once to make them available on other computers.

On macOS, after cloning the repository:

```bash
cd /path/to/spam-email-detection
git pull
python3 start.py
```

On Windows:

```powershell
cd D:\GitHub\spam-email-detection
git pull
python start.py
```

Python must already be installed. Internet access is needed for the first dependency installation. Each computer creates its own `.venv`; do not copy a Windows environment onto a Mac.

## Validation

The integration verifies syntax, input handling, unavailable-model behaviour, and preserved file hashes. Full saved-model inference requires compatible ML dependencies; those packages are absent from the bundled verification runtime. Original result files are evidence supplied by teammates, not newly reproduced results.

## Latest import

Desktop contributions were copied without modifying the originals. See `docs/latest_import.json` for file hashes. Previous differing files are recoverable in `.integration-backups/`. Distinct model-package CSVs are preserved under `data/splits/` and `data/splits/`; do not substitute older processed splits for retraining. Historical files in `results/` and older inventory documents predate this import. The website compares supplied validation metrics; these have not been reproduced by retraining.

## File organization

Current shared CSVs live in `data/splits/`; identical contribution copies have been consolidated. Data preparation materials remain in `data/preparation_submission/`. Each model keeps its code and fitted artifacts together under `src/`. Historical scripts and update notes live in `docs/archive/`; they may refer to removed datasets and are not current entry points. `.integration-backups/` contains recoverable previous files.

## Training from shared data

All training scripts resolve `data/splits/` from their own location, independent of the working directory. CSV columns are `text,label`; 0 is legitimate and 1 is spam.

```powershell
python src/naive_bayes/train_model_1.py
python src/logistic_regression/train_model_2.py
python src/neural_network/train_model_3.py tune
python src/neural_network/train_model_3.py seeds
python src/neural_network/train_model_3.py final
```

Naive Bayes and logistic regression fit on train and select on validation; they do not read test. Neural tuning and seed checks use train/validation; final evaluates test. Running training writes model artifacts; checking data access does not retrain.
