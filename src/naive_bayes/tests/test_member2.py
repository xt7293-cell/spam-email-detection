"""Focused unit checks. These use synthetic fixtures, not the held-out test set."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import decode_email, normalize_email, make_vectorizer, load_split
from predict import predict_emails


class Member2Tests(unittest.TestCase):
    def test_serialized_list_and_plain_text_are_equivalent(self):
        email = "Subject: Meeting\nHello team &amp; colleagues"
        self.assertEqual(normalize_email(repr([email])), normalize_email(email))
        self.assertEqual(decode_email(repr(["Subject: ", "hello\n"])), "Subject: hello\n")

    def test_unicode_preserved_and_malformed_literal_is_plain_text(self):
        self.assertEqual(normalize_email("Café 会议\nAgenda"), "café 会议 agenda")
        unsafe = "[__import__('os').system('echo unsafe')]"
        self.assertEqual(decode_email(unsafe), unsafe)

    def test_validation_words_do_not_enter_vocabulary_or_idf(self):
        vectorizer = make_vectorizer(min_df=1)
        vectorizer.fit(["meeting project agenda", "offer prize reward"])
        vocabulary = dict(vectorizer.vocabulary_)
        idf = vectorizer.idf_.copy()
        vectorizer.transform(["validationonlyword meeting"])
        self.assertNotIn("validationonlyword", vectorizer.vocabulary_)
        self.assertEqual(vocabulary, vectorizer.vocabulary_)
        np.testing.assert_array_equal(idf, vectorizer.idf_)

    def test_save_load_and_adapter_predictions(self):
        model = Pipeline([("tfidf", make_vectorizer(min_df=1)), ("nb", MultinomialNB())])
        model.fit(["project meeting", "project schedule", "free prize", "claim prize"], [0, 0, 1, 1])
        samples = ["project meeting", "free prize"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(model.predict(samples), loaded.predict(samples))
            output = predict_emails(samples, loaded)
            self.assertEqual([item["label"] for item in output], model.predict(samples).tolist())
            np.testing.assert_allclose([item["spam_probability"] for item in output],
                model.predict_proba(samples)[:, 1])
            self.assertEqual(predict_emails(repr([samples[0]]), loaded)[0], output[0])

    def test_empty_input_rejected_and_unseen_tokens_flagged(self):
        model = Pipeline([("tfidf", make_vectorizer(min_df=1)), ("nb", MultinomialNB())])
        model.fit(["project meeting", "free prize"], [0, 1])
        with self.assertRaises(ValueError):
            predict_emails("  \n ", model)
        self.assertTrue(predict_emails("zzzzunknownqqqq", model)[0]["out_of_vocabulary"])
        self.assertEqual(predict_emails([], model), [])

    def test_bad_dataset_fails_instead_of_silent_cleaning(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for contents in ["text,label\nhello,2\n", "text,label\n,0\nspam,1\n",
                             "email,label\nhello,0\nspam,1\n"]:
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_split(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
