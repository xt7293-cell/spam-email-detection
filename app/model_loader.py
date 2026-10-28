"""Inference only. Saved preprocessors and decision rules are preserved."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import warnings
import threading
ROOT = Path(__file__).resolve().parents[1]
SPECS = [('naive_bayes', 'Naive Bayes'), ('logistic_regression', 'Logistic regression'), ('neural_network', 'Neural network')]

class Models:
    def __init__(self):
        self.models = {}
        self.errors = {}
        self.lock = threading.Lock()
        for key, name in SPECS:
            try:
                self.models[key] = self.load(key)
            except Exception as exc:
                self.errors[key] = f'{type(exc).__name__}: {exc}'

    def load(self, key):
        import joblib
        if key == 'naive_bayes':
            folder = ROOT / 'src/naive_bayes'
            # The original pickle references this exact module name.
            spec = importlib.util.spec_from_file_location('preprocessing', folder / 'preprocessing.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules['preprocessing'] = module
            spec.loader.exec_module(module)
            return joblib.load(folder / 'artifacts/naive_bayes_pipeline.joblib')
        if key == 'logistic_regression':
            path = ROOT / 'models/logistic_regression/pipeline.joblib'
            if not path.exists(): raise FileNotFoundError('No trained logistic-regression pipeline was supplied.')
            return joblib.load(path)
        import tensorflow as tf
        folder = ROOT / 'src/neural_network/nn_artifacts'
        vectorizer = joblib.load(folder / 'model/tfidf_vectorizer.joblib')
        model = tf.keras.models.load_model(folder / 'model/spam_nn.keras', compile=False)
        config = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
        return vectorizer, model, float(config['threshold_selected_on_validation'])

    def predict(self, text):
        results = []
        with self.lock:
            for key, name in SPECS:
                result = {'id': key, 'name': name}
                if key not in self.models:
                    result.update(status='unavailable', message=self.errors[key])
                else:
                    start = time.perf_counter()
                    try:
                        if key == 'neural_network':
                            vec, model, threshold = self.models[key]
                            matrix = vec.transform([text])
                            score = float(model(matrix.toarray().astype('float32'), training=False).numpy().ravel()[0])
                            label = int(score >= threshold)
                            empty = matrix.nnz == 0
                            rule = f'Saved neural network; validation-selected threshold {threshold:g}.'
                        else:
                            model = self.models[key]
                            label = int(model.predict([text])[0])
                            if label not in (0, 1): raise ValueError('Expected labels 0 and 1.')
                            empty = False
                            if key == 'naive_bayes': empty = model.named_steps['tfidf'].transform([text]).nnz == 0
                            rule = 'Saved prediction pipeline; original classifier decision rule.'
                        result.update(status='ready', label=label, label_name='Spam' if label else 'Legitimate', milliseconds=round((time.perf_counter()-start)*1000, 2), message=rule, out_of_vocabulary=empty)
                    except Exception as exc:
                        result.update(status='error', message=f'{type(exc).__name__}: {exc}')
                results.append(result)
        return results

    def status(self):
        return [{'id':key,'name':name,'status':'ready' if key in self.models else 'unavailable','message':self.errors.get(key, 'Saved model loaded.')} for key,name in SPECS]
