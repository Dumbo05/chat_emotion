import json,sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,'.')
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb
root=Path('.'); samples=discover_tess(root/'datasets/TESS')+discover_crema(root/'datasets/CREMA-D')+discover_emodb(root/'datasets/EmoDB')
split=json.loads((root/'models/speech/wavlm_clean_split_v2.json').read_text(encoding='utf8'))
assert split['test_status']=='sealed-not-evaluated'
sp=np.array([s.speaker for s in samples]); y=np.array([LABELS.index(s.label) for s in samples]); X=np.load(root/'models/speech/wavlm_clean_v2_layer_stats.npy',mmap_mode='r')
assert len(X)==len(samples)
tr=np.flatnonzero(np.isin(sp,split['train_speakers'])); va=np.flatnonzero(np.isin(sp,split['validation_speakers']))
pools={'mean':slice(0,768),'std':slice(768,1536),'mean_std':slice(0,1536)}; rows=[]
for pool,sl in pools.items():
 for layer in range(13):
  m=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight='balanced',dual='auto',max_iter=5000))
  m.fit(np.asarray(X[tr,layer,sl],np.float32),y[tr]); p=m.predict(np.asarray(X[va,layer,sl],np.float32))
  row={'pooling':pool,'layer':layer,'accuracy':float(accuracy_score(y[va],p)),'macro_f1':float(f1_score(y[va],p,average='macro'))};rows.append(row);print(row,flush=True)
best={pool:max((r for r in rows if r['pooling']==pool),key=lambda r:r['macro_f1']) for pool in pools}
out={'protocol':split['protocol'],'test_status':split['test_status'],'train_samples':len(tr),'validation_samples':len(va),'method':'LinearSVC(C=0.01, balanced) used only as a validation layer probe','rows':rows,'selected_layer_by_pooling':best}
(root/'models/speech/wavlm_clean_v2_layer_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('SELECTED',json.dumps(best,ensure_ascii=False))
