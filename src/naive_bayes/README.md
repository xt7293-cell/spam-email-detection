# Member 2 Submission: Spam Email Detection

TF-IDF and Multinomial Naive Bayes, using Member 1's updated dataset of
28 October 2026. Labels: 0 = legitimate, 1 = spam.

## Contents

| File or folder | Purpose |
| --- | --- |
| `Report.docx` | Project report |
| `preprocessing.py` | Email decoding, normalization and TF-IDF configuration |
| `train_model_1.py` | Train and compare all 32 configurations |
| `predict.py` | Load the saved model and predict new email text |
| `verify_saved_model.py` | Verify input/model hashes and reproduce validation evidence |
| `requirements.txt` | Exact package versions from the recorded environment |
| `tests/test_member2.py` | Six focused tests with synthetic samples |
| `data/` | Unmodified training and validation CSVs |
| `input_manifest.json` | Input file sizes and SHA256 hashes |
| `artifacts/` | Trained pipeline, parameter experiments, predictions, errors, figures and audit evidence |

## Setup and prediction

Use Python 3.12. The recorded run used Python 3.12.14.
Open PowerShell in this folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --text "Subject: Team meeting. Please review the agenda."
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv` and
replace the interpreter path with `.venv/bin/python`.

## Reproduce and verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_saved_model.py
```

To repeat the full 32-setting training search:

```powershell
.\.venv\Scripts\python.exe train_model_1.py
```

Training updates the files in `artifacts/`, including figures. The report is
a separate saved document and does not update automatically.

## Recorded validation results

Training contains 21,460 emails and validation contains 4,390.
The selected configuration uses unigrams, min_df=2, sublinear_tf=False,
max_features=50,000 and MultinomialNB alpha=0.1.

| Validation metric | Result |
| --- | --- |
| Accuracy | 98.9749% |
| Spam precision | 98.9499% |
| Spam recall | 98.9027% |
| Spam F1 | 98.9263% |
| False positives / false negatives | 22 / 23 |

The vocabulary, IDF and classifier are fitted on training only. Validation
spam F1 selects parameters, with precision and experiment ID as tie breakers.
These are validation results. The common final test evaluation belongs to the
group's evaluation stage. The test CSV is not included here.

`validation_results.json` records the exact parameters, metrics, runtime,
environment and training source hashes. `experiments.csv` contains all 32
settings. `validation_predictions.csv` contains every validation prediction,
while `validation_errors.csv` includes the 45 errors and text excerpts.
`indicative_features.csv` supports the report's feature discussion.

## Data and model notes

`data_audit.json` covers training and validation during model training.
`split_integrity_audit.json` preserves a separate all-split integrity check:
the test data were inspected only for counts, labels and text fingerprints,
without model predictions or test metrics.

Under Member 2's list decoding, HTML unescaping, case and whitespace
normalization, 56 extra equivalent rows remain within training, 10 within
validation and 10 within testing. No normalized overlap was found between
splits. The shared CSVs remain unchanged. This does not rule out all semantic
similarity or source bias.

The saved joblib Pipeline includes preprocessing, TF-IDF and Naive Bayes.
Keep `preprocessing.py` available when loading it. For integration:

```python
from predict import load_model, predict_emails

model = load_model()
result = predict_emails(["Subject: Meeting. Please review the agenda."], model)
```

Predicted probabilities are uncalibrated model estimates. Historical
spam/legitimate labels do not establish dedicated phishing detection or
performance on current email, images, attachments or Chinese messages.
