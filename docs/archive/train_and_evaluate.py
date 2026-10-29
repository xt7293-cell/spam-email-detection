"""Train LR using fixed splits and the supplied training-fitted NB vectorizer; final common test evaluation."""
from pathlib import Path
import sys,json,time,hashlib,importlib.util
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[2]
def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2),encoding='utf-8')
spec=importlib.util.spec_from_file_location('preprocessing',ROOT/'src/naive_bayes/preprocessing.py');pre=importlib.util.module_from_spec(spec);sys.modules['preprocessing']=pre;spec.loader.exec_module(pre)
# Verify the feature-fitting source matches the preserved training/validation inputs.
audit=json.loads((ROOT/'src/naive_bayes/artifacts/data_audit.json').read_text())
for split in ['train','validation']:
 assert hashlib.sha256((ROOT/f'data/processed/{split}.csv').read_bytes()).hexdigest()==audit['splits'][split]['sha256']
frames={s:pd.read_csv(ROOT/f'data/processed/{s}.csv',encoding='utf-8-sig',keep_default_na=False) for s in ['train','validation']}
nb=joblib.load(ROOT/'src/naive_bayes/artifacts/naive_bayes_pipeline.joblib');vec=nb.named_steps['tfidf']
X=vec.transform(frames['train'].text);V=vec.transform(frames['validation'].text)
def metrics(y,p):
 tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel());return {'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1':float(f1_score(y,p,zero_division=0)),'false_positive_rate':fp/(fp+tn),'tn':tn,'fp':fp,'fn':fn,'tp':tp,'n':len(y)}
experiments=[];best=None
for c in [.1,1.,10.]:
 clf=LogisticRegression(C=c,solver='liblinear',max_iter=1000,random_state=42)
 started=time.perf_counter();clf.fit(X,frames['train'].label);scores=metrics(frames['validation'].label,clf.predict(V));row={'C':c,'fit_seconds':time.perf_counter()-started,'converged':bool(np.all(clf.n_iter_<1000)),**scores};experiments.append(row)
 if best is None or (row['f1'],row['precision'],-c)>(best[0]['f1'],best[0]['precision'],-best[0]['C']):best=(row,clf)
 print('Validation C',c,'F1',row['f1'],flush=True)
assert all(e['converged'] for e in experiments)
lr=Pipeline([('tfidf',vec),('lr',best[1])]);path=OUT/'models/logistic_regression/pipeline.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(lr,path)
assert np.array_equal(lr.predict(frames['validation'].text[:20]),joblib.load(path).predict(frames['validation'].text[:20]))
save('results/logistic_regression_validation.json',{'model':'Logistic regression','selected':best[0],'experiments':experiments,'selection':'Highest validation spam F1, then precision, then smaller C','tfidf':'Reused original NB training-fitted vectorizer; no refit','seed':42,'solver':'liblinear','max_iter':1000,'training_sha256':audit['splits']['train']['sha256']})
# Test opens only after configuration selection; no tuning follows.
frames['test']=pd.read_csv(ROOT/'data/processed/test.csv',encoding='utf-8-sig',keep_default_na=False)
import tensorflow as tf
folder=ROOT/'src/neural_network/nn_artifacts';nv=joblib.load(folder/'model/tfidf_vectorizer.joblib');nn=tf.keras.models.load_model(folder/'model/spam_nn.keras',compile=False);nconfig=json.loads((folder/'results.json').read_text());threshold=float(nconfig['threshold_selected_on_validation'])
text=frames['test'].text;y=frames['test'].label
common={'split':'Test','rows':len(y),'models':{},'test_sha256':hashlib.sha256((ROOT/'data/processed/test.csv').read_bytes()).hexdigest(),'note':'Final fixed-model evaluation on the same preserved test CSV. LR selected on validation only. Existing neural test results had already been viewed; this is not a fresh unseen holdout for the whole project.'}
for key,model in [('naive_bayes',nb),('logistic_regression',lr)]:common['models'][key]=metrics(y,model.predict(text))
prob=[]
for start in range(0,len(text),64):prob.extend(nn(nv.transform(text.iloc[start:start+64]).toarray().astype('float32'),training=False).numpy().ravel().tolist())
common['models']['neural_network']=metrics(y,(np.asarray(prob)>=threshold).astype(int))
sets={s:set(f.text.map(pre.normalize_email)) for s,f in frames.items()}
common['normalised_overlap']={'train_validation_unique':len(sets['train']&sets['validation']),'train_test_unique':len(sets['train']&sets['test']),'validation_test_unique':len(sets['validation']&sets['test'])}
save('results/common_test_results.json',common)
counts={s:{'rows':len(f),'legitimate':int((f.label==0).sum()),'spam':int((f.label==1).sum())} for s,f in frames.items()};save('results/dataset_summary.json',{'name':'Enron-Spam email collection (supplied CSVs)','total':sum(x['rows'] for x in counts.values()),'splits':counts,'source_note':'Dataset source documentation supplied in docs/contributions/enron_dataset_source_and_overview.docx; original download URL not independently verified.','overlap':common['normalised_overlap'],'near_duplicate_audit':'Not exhaustive; original fixed splits retained.'})
print(json.dumps(common,indent=2),flush=True)
