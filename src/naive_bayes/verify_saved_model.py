"""Independently verify the delivered artifacts using train/validation only."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import decode_email, load_split, make_vectorizer
from train_model_1 import metrics, sha256, write_json
from predict import predict_emails

ROOT = Path(__file__).resolve().parent


def main():
    artifact = ROOT / "artifacts"
    result = json.loads((artifact / "validation_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest.items():
        assert sha256(ROOT / "data" / name) == record["sha256"], name
    for name, digest in result["source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    model_file = artifact / "naive_bayes_pipeline.joblib"
    assert sha256(model_file) == result["pipeline_sha256"]
    model = joblib.load(model_file)
    train = load_split(ROOT / "data" / "train.csv")
    val = load_split(ROOT / "data" / "validation.csv")
    prediction = model.predict(val.text)
    probability = model.predict_proba(val.text)[:, 1]
    reproduced = metrics(val.label, prediction, probability)
    for name, value in reproduced.items():
        np.testing.assert_allclose(value, result["selected"][name], rtol=0, atol=1e-12)
    saved_predictions = pd.read_csv(artifact / "validation_predictions.csv")
    np.testing.assert_array_equal(prediction, saved_predictions.predicted_label)
    np.testing.assert_allclose(probability, saved_predictions.spam_probability, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(prediction, model.predict(val.text.map(decode_email)))
    np.testing.assert_allclose(probability, model.predict_proba(val.text.map(decode_email))[:, 1], rtol=0, atol=0)
    config = result["tfidf_config"]
    independently_fitted = make_vectorizer(**config).fit(train.text)
    saved_vectorizer = model.named_steps["tfidf"]
    assert independently_fitted.vocabulary_ == saved_vectorizer.vocabulary_
    np.testing.assert_array_equal(independently_fitted.idf_, saved_vectorizer.idf_)
    np.testing.assert_array_equal(model.named_steps["nb"].class_count_,
        [(train.label == 0).sum(), (train.label == 1).sum()])
    api_output = predict_emails(val.text.iloc[:30], model)
    np.testing.assert_array_equal([x["label"] for x in api_output], prediction[:30])
    np.testing.assert_allclose([x["spam_probability"] for x in api_output], probability[:30], rtol=0, atol=0)
    verification = {
        "passed": True, "all_input_file_hashes_match": True,
        "training_source_hashes_match": True, "model_hash_matches": True,
        "validation_metrics_independently_reproduced": reproduced,
        "all_validation_predictions_match_saved_csv": True,
        "all_validation_wrapped_and_plain_inputs_match": True,
        "vocabulary_and_idf_match_independent_training_only_fit": True,
        "shared_vectorizer_available_as_pipeline_tfidf_step": True,
        "classifier_class_counts_match_training_only": True,
        "prediction_adapter_matches_pipeline": True,
        "test_set_usage": "Not included in this Member 2 package; never read or evaluated",
    }
    write_json(artifact / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
