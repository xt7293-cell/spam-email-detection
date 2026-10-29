"""Fast tests with synthetic text; do not read the dataset or held-out test split."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import make_vectorizer, normalize_email
from predict_model_2 import predict_emails


class TestMember3(unittest.TestCase):
    def test_text_normalization(self):
        self.assertEqual(normalize_email(repr(['Subject: Hi &amp; Bye\n'])), 'subject: hi & bye')

    def test_train_only_features_and_prediction(self):
        train = ['project meeting tomorrow', 'weekly agenda update',
                 'claim free prize now', 'win your free prize']
        y = [0, 0, 1, 1]
        vec = make_vectorizer(min_df=1)
        x = vec.fit_transform(train)
        before = vec.idf_.copy()
        val = vec.transform(['unseenword free prize'])
        self.assertNotIn('unseenword', vec.vocabulary_)
        np.testing.assert_equal(before, vec.idf_)
        self.assertEqual(val.shape[1], x.shape[1])
        lr = LogisticRegression(solver='liblinear', random_state=42)
        lr.fit(x, y)
        full = Pipeline([('tfidf', vec), ('lr', lr)])
        p = predict_emails('claim free prize', full)[0]
        self.assertEqual(p['label'], 1)
        self.assertGreater(p['spam_probability'], 0.5)

    def test_reject_empty_input(self):
        with self.assertRaises(ValueError):
            predict_emails('   ')


if __name__ == '__main__':
    unittest.main()
