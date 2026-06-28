import json,sys
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,'.')
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb
root=Path('.');samples=discover_tess(root/'datasets/TESS')+discover_crema(root/'datasets/CREMA-D')+discover_emodb(root/'datasets/EmoDB');split=json.loads((root/'models/speech/wavlm_clean_split_v2.json').read_text(encoding='utf8'));assert split['test_status']=='sealed-not-evaluated';scan=json.loads((root/'models/speech/wavlm_clean_v2_rbf_layer_scan.json').read_text(encoding='utf8'))
sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);X=np.load(root/'models/speech/wavlm_clean_v2_layer_stats.npy',mmap_mode='r');tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['validation_speakers']));slices={'mean':slice(0,768),'std':slice(768,1536),'mean_std':slice(0,1536)}
Cs=[2.,3.,5.];gfs=[.25,.5,.75];configs=list(scan['best'])
prepared={}
for key in configs:
 c=scan['best'][key];sl=slices[c['pooling']];sc=StandardScaler();prepared[key]=(c,sc.fit_transform(np.asarray(X[tr,c['layer'],sl],np.float32)),sc.transform(np.asarray(X[va,c['layer'],sl],np.float32)))
def one(key,C,gf):
 c,Xt,Xv=prepared[key];m=SVC(C=C,gamma=gf/Xt.shape[1],class_weight=c['class_weight'],cache_size=1024);m.fit(Xt,y[tr]);p=m.predict(Xv);return {'config':key,'pooling':c['pooling'],'layer':c['layer'],'C':C,'gamma_factor':gf,'gamma':gf/Xt.shape[1],'class_weight':c['class_weight'],'accuracy':float(accuracy_score(y[va],p)),'macro_f1':float(f1_score(y[va],p,average='macro'))}
rows=Parallel(n_jobs=4,prefer='threads')(delayed(one)(k,C,g) for k in configs for C in Cs for g in gfs);best={k:max((r for r in rows if r['config']==k),key=lambda r:r['macro_f1']) for k in configs};out={'protocol':split['protocol'],'test_status':split['test_status'],'grid':{'C':Cs,'gamma_factor_over_n_features':gfs},'rows':rows,'frozen_wavlm_configs':best};(root/'models/speech/wavlm_clean_v2_frozen_configs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(best,ensure_ascii=False,indent=2))
