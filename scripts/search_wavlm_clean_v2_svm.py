import json,sys,time
from pathlib import Path
import joblib,numpy as np
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,'.')
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb
root=Path('.'); samples=discover_tess(root/'datasets/TESS')+discover_crema(root/'datasets/CREMA-D')+discover_emodb(root/'datasets/EmoDB')
split=json.loads((root/'models/speech/wavlm_clean_split_v2.json').read_text(encoding='utf8')); assert split['test_status']=='sealed-not-evaluated'
probe=json.loads((root/'models/speech/wavlm_clean_v2_layer_probe.json').read_text(encoding='utf8'))
sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);X=np.load(root/'models/speech/wavlm_clean_v2_layer_stats.npy',mmap_mode='r')
tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['validation_speakers']))
pools={'mean':slice(0,768),'std':slice(768,1536),'mean_std':slice(0,1536)}
Cs=[.5,1.,3.,10.]; gamma_factors=[.5,1.,2.]; rows=[]
for pool,sl in pools.items():
 layer=probe['selected_layer_by_pooling'][pool]['layer']; scaler=StandardScaler(); Xt=scaler.fit_transform(np.asarray(X[tr,layer,sl],np.float32)); Xv=scaler.transform(np.asarray(X[va,layer,sl],np.float32)); dim=Xt.shape[1]
 def fit_one(C,gf,balanced):
  m=SVC(C=C,gamma=gf/dim,kernel='rbf',class_weight='balanced' if balanced else None,cache_size=1024)
  m.fit(Xt,y[tr]);p=m.predict(Xv)
  return {'pooling':pool,'layer':layer,'C':C,'gamma_factor':gf,'gamma':gf/dim,'class_weight':'balanced' if balanced else None,'accuracy':float(accuracy_score(y[va],p)),'macro_f1':float(f1_score(y[va],p,average='macro'))}
 tasks=[(C,gf,b) for C in Cs for gf in gamma_factors for b in (False,True)]
 result=Parallel(n_jobs=4,prefer='threads')(delayed(fit_one)(*task) for task in tasks);rows+=result
 for row in sorted(result,key=lambda r:r['macro_f1'],reverse=True)[:5]: print(row,flush=True)
best={}
for pool in pools:
 for weight in (None,'balanced'):
  key=f'{pool}_{weight or "unbalanced"}';best[key]=max((r for r in rows if r['pooling']==pool and r['class_weight']==weight),key=lambda r:r['macro_f1'])
out={'protocol':split['protocol'],'test_status':split['test_status'],'grid':{'C':Cs,'gamma_factor_over_n_features':gamma_factors,'class_weight':[None,'balanced']},'rows':rows,'best_by_pooling_and_weight':best}
(root/'models/speech/wavlm_clean_v2_validation_search.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('BEST',json.dumps(best,ensure_ascii=False,indent=2))
