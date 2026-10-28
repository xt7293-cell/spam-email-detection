"""Read-only experiment evidence. No model fitting or new test evaluation."""
import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def excerpt(text):
    text = re.sub(r'[\w.+-]+\s*@\s*[\w.-]+', '[email redacted]', text or '')
    return text[:420]

def metrics(values, split):
    return {'split': split, 'metrics': {k: values[k] for k in ('accuracy','precision','recall','f1','false_positive_rate')}, 'matrix': [[values['tn'],values['fp']],[values['fn'],values['tp']]]}

def cases(folder, neural, threshold, split):
    path = folder / ('predictions/'+split+'_predictions.csv' if neural else 'validation_predictions.csv')
    error_path = folder / ('predictions/'+split+'_misclassified.csv' if neural else 'validation_errors.csv')
    errors = {}
    if error_path.exists():
        with error_path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f): errors[str(r['row_id' if neural else 'row_index_0based'])] = r
    near = []; mistakes = []
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            row_id = r['row_id' if neural else 'row_index_0based']
            prob = float(r['prob_spam' if neural else 'spam_probability'])
            actual = int(r['label' if neural else 'true_label']); pred = int(r['pred' if neural else 'predicted_label'])
            if not math.isfinite(prob): continue
            detail = errors.get(row_id, r)
            item = {'row':int(row_id),'probability':prob,'actual':actual,'predicted':pred,'distance':abs(prob-threshold),'excerpt':excerpt(detail.get('text_preview',detail.get('text_excerpt','')))}
            near.append(item)
            if actual != pred: mistakes.append(item)
    near.sort(key=lambda r:r['distance'])
    # Resolve missing marginal excerpts by saved row index, without inference.
    missing={r['row']:r for r in near[:5] if not r['excerpt']}
    source=ROOT/'data/processed'/f'{split}.csv'
    if missing and source.exists():
        with source.open(encoding='utf-8-sig',newline='') as f:
            for i,r in enumerate(csv.DictReader(f)):
                if i in missing: missing[i]['excerpt']=excerpt(r['text'])
                if i>max(missing): break
    return {'split':split.title(),'threshold':threshold,'band':0.05,'marginal_count':sum(r['distance']<=0.05 for r in near),'nearest':near[:5],'errors':sorted(mistakes,key=lambda r:r['distance'])[:5],'source':str(path.relative_to(ROOT))}

@lru_cache(maxsize=1)
def experiment_details():
    output=[]
    for key,name in [('naive_bayes','Naive Bayes'),('logistic_regression','Logistic regression'),('neural_network','Neural network')]:
        item={'id':key,'name':name}
        try:
            if key=='logistic_regression':
                item.update(status='unavailable',message='No training report or saved evaluation was supplied.'); output.append(item); continue
            neural=key=='neural_network'
            folder=ROOT/('src/neural_network/nn_artifacts' if neural else 'src/naive_bayes/artifacts')
            source=folder/('results.json' if neural else 'validation_results.json')
            d=read_json(source)
            if neural:
                threshold=float(d['threshold_selected_on_validation'])
                item['settings']={'Preprocessing':'Saved word TF-IDF: lowercase and Unicode accent stripping; no stop-word list; no Naive Bayes wrapper decoder.','TF-IDF':json.dumps({k:d['config'][k] for k in ('max_features','ngram_range','sublinear_tf','min_df')}),'Training':json.dumps(d['config']),'Training outcome':json.dumps(d['training']),'Random seed':str(d['seed']),'Decision rule':f'Spam if estimated spam probability >= {threshold:g}.','Tuning':'Early stopping monitors validation loss (patience 3, restore best weights). Decision threshold selected using validation results. See saved threshold sweep for candidates.'}
                item['evaluations']=[metrics(d[s],s.title()) for s in ('validation','test')]
            else:
                threshold=0.5
                item['settings']={'Preprocessing':'Decode serialized list wrappers; HTML entity decoding; lowercase; collapse whitespace. Word tokens have at least two characters; no stemming or stop-word removal.','TF-IDF':json.dumps(d['tfidf_config'])+'; max_df=1.0, smoothed IDF, L2 normalisation.','Training':f"Multinomial Naive Bayes; alpha={d['selected']['alpha']}; fitted on training data only.",'Tuning':f"{d['number_of_configurations']} configurations. Selection: {d['selection_rule']}.",'Decision rule':'Classifier argmax. With two classes the boundary is spam probability 0.5; an exact tie resolves to legitimate.'}
                item['evaluations']=[metrics(d['selected'],'Validation')]
            item.update(status='ready',source=str(source.relative_to(ROOT)),cases=cases(folder,neural,threshold,'validation'))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            item.update(status='error',message=f'Saved evidence could not be read: {exc}')
        output.append(item)
    return {'models':output,'note':'Saved experiment evidence, not newly reproduced results. Validation/test scores are separate. The models use different fitted TF-IDF configurations, so results reflect both classifier and feature choices. Marginal means within 0.05 probability units of that model’s threshold; this is a descriptive analysis rule, not a calibrated confidence guarantee. Examples use zero-based dataset row IDs and show truncated, email-address-redacted text.'}
