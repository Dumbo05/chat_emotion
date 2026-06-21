import json, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0,'.')
from scripts.train_simsan import LABEL_TO_ID, discover_tess, discover_crema, discover_emodb
samples=discover_tess(Path('datasets/TESS'))+discover_crema(Path('datasets/CREMA-D'))+discover_emodb(Path('datasets/EmoDB'))
split=json.loads(Path('models/speech/simsan_split_manifest.json').read_text(encoding='utf8'))
sp=np.array([s.speaker for s in samples]); y=np.array([LABEL_TO_ID[s.label] for s in samples]); X=np.load('models/speech/wavlm_layer_stats.npy',mmap_mode='r')
tr=np.flatnonzero(np.isin(sp,split['train_speakers'])); va=np.flatnonzero(np.isin(sp,split['validation_speakers']))
linear=[]
for layer in range(X.shape[1]):
 m=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight='balanced',dual='auto',max_iter=5000))
 m.fit(np.asarray(X[tr,layer],np.float32),y[tr]); p=m.predict(np.asarray(X[va,layer],np.float32)); row=(layer,accuracy_score(y[va],p),f1_score(y[va],p,average='macro'));linear.append(row);print('linear',row,flush=True)
top=[x[0] for x in sorted(linear,key=lambda x:x[2],reverse=True)[:4]]
best=(-1,None,None)
for layer in top:
 for C in [.3,1,3,10]:
  m=make_pipeline(StandardScaler(),SVC(C=C,gamma='scale',class_weight='balanced',probability=True,cache_size=4096))
  m.fit(np.asarray(X[tr,layer],np.float32),y[tr]); p=m.predict(np.asarray(X[va,layer],np.float32)); row=(layer,C,accuracy_score(y[va],p),f1_score(y[va],p,average='macro'));print('rbf',row,flush=True)
  if row[3]>best[0]: best=(row[3],row,m)
print('BEST',best[:2]);joblib.dump({'model':best[2],'layer':best[1][0],'validation_accuracy':best[1][2],'validation_f1_macro':best[1][3]},'models/speech/wavlm_simsan_head.joblib',compress=3)
Path('models/speech/wavlm_validation_search.json').write_text(json.dumps({'linear':linear,'best':best[1]},indent=2),encoding='utf8')
