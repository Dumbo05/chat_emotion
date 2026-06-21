import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
sys.path.insert(0,'.')
from scripts.train_simsan import LABELS,LABEL_TO_ID,discover_tess,discover_crema,discover_emodb
samples=discover_tess(Path('datasets/TESS'))+discover_crema(Path('datasets/CREMA-D'))+discover_emodb(Path('datasets/EmoDB'))
split=json.loads(Path('models/speech/simsan_split_manifest.json').read_text(encoding='utf8'))
sp=np.array([s.speaker for s in samples]);y=np.array([LABEL_TO_ID[s.label] for s in samples]);X=np.load('models/speech/wavlm_layer_stats.npy',mmap_mode='r')
tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['validation_speakers']))
m=make_pipeline(StandardScaler(),SVC(C=3,gamma='scale',class_weight='balanced',probability=False,cache_size=4096))
m.fit(np.asarray(X[tr,10],np.float32),y[tr]);p=m.predict(np.asarray(X[va,10],np.float32))
report=classification_report(y[va],p,labels=range(len(LABELS)),target_names=LABELS,output_dict=True,zero_division=0)
out={'split':'locked speaker-independent validation','samples':int(len(va)),'accuracy':float(accuracy_score(y[va],p)),'macro_f1':float(f1_score(y[va],p,average='macro')),'classification_report':report,'confusion_matrix':confusion_matrix(y[va],p,labels=range(len(LABELS))).tolist()}
Path('models/speech/wavlm_simsan_validation_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
joblib.dump({'model':m,'layer':10,'labels':LABELS,'validation':out},'models/speech/wavlm_simsan_head.joblib',compress=3)
print(json.dumps({'accuracy':out['accuracy'],'macro_f1':out['macro_f1'],'per_class':{k:{'recall':report[k]['recall'],'precision':report[k]['precision'],'f1':report[k]['f1-score'],'support':int(report[k]['support'])} for k in LABELS}},ensure_ascii=False,indent=2))
