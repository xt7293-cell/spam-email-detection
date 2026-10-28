"""Aggregate website data only. No dataset message text is returned."""
from pathlib import Path
import json
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]
SOURCE_ROOT=ROOT

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pack(values,split):
 return {'split':split,'metrics':{k:values[k] for k in ('accuracy','precision','recall','f1','false_positive_rate')},'matrix':[[values['tn'],values['fp']],[values['fn'],values['tp']]]}
@lru_cache(maxsize=1)
def website_summary():
 common=read(ROOT/'results/common_test_results.json');dataset=read(ROOT/'results/dataset_summary.json')
 dataset.update(source_url='https://huggingface.co/datasets/bvk/ENRON-spam',source_note='Source listed in the supplied dataset documentation.',version_note='Current model split: 30,482 emails. A newer 30,202-email cleaned split exists in preparation_submission; retrain all models before switching.',audit_note='1 normalised train/validation overlap and 3 train/test overlaps. Near-duplicate audit of this older split is not exhaustive.')
 nb=read(SOURCE_ROOT/'src/naive_bayes/artifacts/validation_results.json');nn=read(SOURCE_ROOT/'src/neural_network/nn_artifacts/results.json')
 models=[]
 for key,name,val,threshold,about in [
 ('naive_bayes','Naive Bayes',nb['selected'],.5,{'Method':'Multinomial Naive Bayes','Text':'Decode list wrappers and HTML; lowercase; normalise whitespace.','TF-IDF':str(nb['tfidf_config']),'Training':'Training only; alpha='+str(nb['selected']['alpha']),'Tuning':str(nb['number_of_configurations'])+' validation configurations; highest spam F1.'}),
 ('neural_network','Neural network',nn['validation'],float(nn['threshold_selected_on_validation']),{'Method':'Dense neural network','Text':'Lowercase and Unicode accent stripping; own saved vectorizer.','TF-IDF':str({k:nn['config'][k] for k in ['max_features','ngram_range','min_df','sublinear_tf']}),'Training':nn['config']['architecture']+'; '+nn['config']['optimizer'],'Tuning':'Validation early stopping (patience 3); selected threshold 0.7.'})]:
  models.append({'id':key,'name':name,'status':'ready','threshold':threshold,'about':about,'evaluations':[pack(val,'Validation'),pack(common['models'][key],'Test')]})
 models.insert(1,{'id':'logistic_regression','name':'Logistic regression','status':'unavailable','message':'Awaiting your supplied model.','about':{'Status':'Awaiting your supplied model.'},'evaluations':[]})
 return {'models':models,'dataset':dataset,'comparison_note':common['note'],'comparison_split':'Test','comparison_rows':common['rows']}
