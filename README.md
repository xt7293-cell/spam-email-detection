# Spam Email Detection

Two-page email comparison website connected to the supplied saved models. Labels: 0 = legitimate, 1 = spam. Nothing trains when you start the website.

## Start locally (Windows)

Use a Python version supported by your selected TensorFlow release. Create a virtual environment, activate it, then install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app/app.py
```

Open http://127.0.0.1:8000. To enable the neural network, additionally install `requirements-neural.txt` before restarting the app. Errors and absent dependencies appear in the affected model card; the other models remain usable. The server listens only on your laptop. The frontend must be served by this Python application, not opened as a file or deployed alone to GitHub Pages.

## Important compatibility limits

The submitted artifacts report scikit-learn 1.9.1, NumPy 2.5.4, and TensorFlow 2.22.0-rc0. The original Naive Bayes requirements and neural results.json are preserved. The root requirements are broad installation guidance, not a verified lockfile. Prefer matching the original training environment. Cross-version pickle loading may fail or produce warnings; check predictions in the original environment before reporting results. Only load model files you trust.

## Model status

- Naive Bayes: saved pipeline and custom `preprocessing.py` retained together under `src/naive_bayes`.
- Logistic regression: not supplied. Add a trusted fitted text-to-label scikit-learn pipeline as `models/logistic_regression/pipeline.joblib`; restart the app. It must output integer 0/1 labels. No fallback prediction is invented.
- Neural network: original Keras model and TF-IDF under `src/neural_network/nn_artifacts/model`. The saved validation-selected threshold is 0.7; loaded from results.json.

Each model retains its own fitted text processor. The Naive Bayes and neural network use different TF-IDF configurations; these supplied experiments are not a controlled shared-vectorizer comparison. Do not refit a vectorizer during inference. Subject and body are joined with a newline; use complete raw email text with an empty subject if matching the dataset format requires it.

## Layout

- `frontend/`: website input and comparison views.
- `app/`: local HTTP server and model adapters.
- `src/naive_bayes/`: original contribution, inference code, training code, tests, results and saved model.
- `src/neural_network/`: original neural contribution, code, results and saved artifacts.
- `src/logistic_regression/`: instructions for the missing model.
- `data/processed/`: supplied model-package train, validation and test files, centralised without modification.
- `data/preparation_submission/`: separate preparation submission and its different CSV splits.
- `docs/contributions/`: original contribution documents.
- `docs/data_inventory.json`: source hashes, row counts and class counts.
- `docs/archive/`: earlier repository README preserved for reference.

## Data mismatch

The preparation-submission CSVs differ from the model-package CSVs. Preserve both; do not substitute one for the other or claim all experiments used the preparation-submission split. The inventory records both. The supplied neural package already contains test results; this integration does not rerun test evaluation or tune models.

To rerun Naive Bayes deliberately, use:

```
python src/naive_bayes/train_model_1.py --data-dir data/processed --output-dir src/naive_bayes/artifacts
```

The neural training script is preserved, including its original relative paths. Adapt DATA_DIR and ARTIFACTS_DIR before deliberately retraining; do not import the training script from the web application because its training runs at module scope.

## GitHub

Large datasets and model binaries are ignored by `.gitignore` and remain local. This means a fresh Git clone needs the documented artifacts restored separately before inference. Do not commit real private email content. The neural model is approximately 62 MB; use a suitable artifact distribution method if sharing it. No Git push is performed by the installer.

## Validation

The integration verifies syntax, input handling, unavailable-model behaviour, and preserved file hashes. Full saved-model inference requires compatible ML dependencies; those packages are absent from the bundled verification runtime. Original result files are evidence supplied by teammates, not newly reproduced results.
