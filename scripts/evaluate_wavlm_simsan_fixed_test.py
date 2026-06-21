import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
sys.path.insert(0,'.')
from scripts.train_simsan import LABELS,LABEL_TO_ID,discover_tess,discover_crema,discover_emodb
samples=discover_tess(Path('datasets/TESS'))+discover_crema(Path('datasets/CREMA-D'))+discover_emodb(Path('datasets/EmoDB'))
split=json.loads(Path('models/speech/simsan_split_manifest.json').read_text(encoding='utf8'))
sp=np.array([s.speaker for s in samples]); y=np.array([LABEL_TO_ID[s.label] for s in samples]); X=np.load('models/speech/wavlm_layer_stats.npy',mmap_mode='r')
te=np.flatnonzero(np.isin(sp,split['final_test_speakers']))
bundle=joblib.load('models/speech/wavlm_simsan_head.joblib'); p=bundle['model'].predict(np.asarray(X[te,bundle['layer']],np.float32)); used=sorted(np.unique(y[te]).tolist())
rep=classification_report(y[te],p,labels=used,target_names=[LABELS[i] for i in used],output_dict=True,zero_division=0)
out={'status':'fixed-test re-evaluation after test was previously opened by an earlier model; not a pristine blind test','samples':int(len(te)),'test_speakers':split['final_test_speakers'],'labels_evaluated':[LABELS[i] for i in used],'accuracy':float(accuracy_score(y[te],p)),'precision_macro':float(precision_score(y[te],p,labels=used,average='macro',zero_division=0)),'recall_macro':float(recall_score(y[te],p,labels=used,average='macro',zero_division=0)),'f1_macro':float(f1_score(y[te],p,labels=used,average='macro',zero_division=0)),'classification_report':rep,'confusion_matrix':confusion_matrix(y[te],p,labels=used).tolist(),'note':'Surprise excluded: no untouched surprise speaker exists in supplied datasets.'}
Path('models/speech/wavlm_simsan_fixed_test_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'samples':out['samples'],'accuracy':out['accuracy'],'precision_macro':out['precision_macro'],'recall_macro':out['recall_macro'],'f1_macro':out['f1_macro'],'per_class_recall':{k:rep[k]['recall'] for k in out['labels_evaluated']}},ensure_ascii=False,indent=2))
