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
        if key in ('naive_bayes', 'logistic_regression'):
            folder = ROOT / 'src' / key
            # The original pickle references this exact module name.
            spec = importlib.util.spec_from_file_location('preprocessing', folder / 'preprocessing.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules['preprocessing'] = module
            spec.loader.exec_module(module)
            filename = 'naive_bayes_pipeline.joblib' if key == 'naive_bayes' else 'logistic_regression_pipeline.joblib'
            return joblib.load(folder / 'artifacts' / filename)
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
                result = {'id': key, 'name': name, 'evaluations': saved_evaluations(key)}
                if key not in self.models:
                    result.update(status='unavailable', message=self.errors[key])
                else:
                    start = time.perf_counter()
                    try:
                        score = None
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
                            if hasattr(model, 'predict_proba'):
                                classes = list(model.classes_)
                                score = float(model.predict_proba([text])[0][classes.index(1)])
                            empty = False
                            if key == 'naive_bayes': empty = model.named_steps['tfidf'].transform([text]).nnz == 0
                            rule = 'Saved prediction pipeline; original classifier decision rule.'
                        result.update(spam_probability=score, status='ready', label=label, label_name='Spam' if label else 'Legitimate', milliseconds=round((time.perf_counter()-start)*1000, 2), message=rule, out_of_vocabulary=empty)
                        try:
                            result['explanation'] = sentence_influence(self.models[key], key, text, score)
                        except Exception:
                            result['explanation'] = {'status':'unavailable', 'message':'Sentence explanation unavailable.'}
                    except Exception as exc:
                        result.update(status='error', message=f'{type(exc).__name__}: {exc}')
                results.append(result)
        return results

    def status(self):
        return [{'id':key,'name':name,'status':'ready' if key in self.models else 'unavailable','message':self.errors.get(key, 'Saved model loaded.')} for key,name in SPECS]


def sentence_influence(loaded, key, text, full_score, limit=40):
    """Leave-one-sentence-out sensitivity on this input; no fitting or causal claim."""
    import re
    import math
    if full_score is None or not math.isfinite(full_score):
        return {'status':'unavailable', 'message':'This model has no probability score.'}
    spans = []
    start = 0
    for boundary in re.finditer(r'(?<=[.!?])\s+|\n+', text):
        if text[start:boundary.start()].strip(): spans.append((start,boundary.start()))
        start = boundary.end()
    if text[start:].strip(): spans.append((start,len(text)))
    selected = spans[:limit]
    masked = [text[:start]+text[end:] for start,end in selected]
    scores = []
    for offset in range(0,len(masked),8):
        batch = masked[offset:offset+8]
        if key == 'neural_network':
            vectorizer, model, threshold = loaded
            array = vectorizer.transform(batch).toarray().astype('float32')
            scores.extend(float(p) for p in model(array,training=False).numpy().ravel())
        else:
            classes = list(loaded.classes_)
            column = classes.index(1)
            scores.extend(float(row[column]) for row in loaded.predict_proba(batch))
    threshold = float(loaded[2]) if key == 'neural_network' else 0.5
    sentences=[]
    for (start,end),score in zip(selected,scores):
        if not math.isfinite(score): continue
        sentences.append({'start':start,'end':end,'text':text[start:end], 'delta':float(full_score-score),'without_score':score})
    return {'status':'ready','method':'sentence_removal','text':text,'threshold':threshold, 'sentences':sentences,'total_sentences':len(spans),'tested_sentences':len(sentences)}


def saved_evaluations(key):
    from website_summary import website_summary
    model=next((m for m in website_summary()['models'] if m['id']==key), None)
    return model['evaluations'] if model else []
