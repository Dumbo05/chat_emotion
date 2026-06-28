import json,sys
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,'.')
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb
root=Path('.');samples=discover_tess(root/'datasets/TESS')+discover_crema(root/'datasets/CREMA-D')+discover_emodb(root/'datasets/EmoDB')
split=json.loads((root/'models/speech/wavlm_clean_split_v2.json').read_text(encoding='utf8'));assert split['test_status']=='sealed-not-evaluated'
search=json.loads((root/'models/speech/wavlm_clean_v2_validation_search.json').read_text(encoding='utf8'))
sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);X=np.load(root/'models/speech/wavlm_clean_v2_layer_stats.npy',mmap_mode='r');tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['validation_speakers']))
configs=['mean_unbalanced','mean_balanced','std_balanced','mean_std_unbalanced','mean_std_balanced'];slices={'mean':slice(0,768),'std':slice(768,1536),'mean_std':slice(0,1536)}
def one(key,layer):
 c=search['best_by_pooling_and_weight'][key];sl=slices[c['pooling']];sc=StandardScaler();Xt=sc.fit_transform(np.asarray(X[tr,layer,sl],np.float32));Xv=sc.transform(np.asarray(X[va,layer,sl],np.float32));m=SVC(C=c['C'],gamma=c['gamma_factor']/Xt.shape[1],class_weight=c['class_weight'],cache_size=1024);m.fit(Xt,y[tr]);p=m.predict(Xv);return {'config':key,'pooling':c['pooling'],'layer':layer,'C':c['C'],'gamma_factor':c['gamma_factor'],'class_weight':c['class_weight'],'accuracy':float(accuracy_score(y[va],p)),'macro_f1':float(f1_score(y[va],p,average='macro'))}
rows=Parallel(n_jobs=4,prefer='threads')(delayed(one)(key,l) for key in configs for l in range(13));best={key:max((r for r in rows if r['config']==key),key=lambda r:r['macro_f1']) for key in configs}
(root/'models/speech/wavlm_clean_v2_rbf_layer_scan.json').write_text(json.dumps({'protocol':split['protocol'],'test_status':split['test_status'],'rows':rows,'best':best},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(best,ensure_ascii=False,indent=2))
