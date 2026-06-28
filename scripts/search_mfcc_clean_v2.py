import json,sys
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,'.')
from emotion_app.audio_features import select_speaker_reduced_features
root=Path('.');split=json.loads((root/'models/speech/wavlm_clean_split_v2.json').read_text(encoding='utf8'));assert split['test_status']=='sealed-not-evaluated';d=np.load(root/'models/speech/multidataset_features.npz');sp=d['speakers'];labels=['anger','disgust','fear','joy','sadness','surprise','neutral'];y=np.array([labels.index(v) for v in d['y']]);base=d['X'];features={'full_177':base,'speaker_reduced_129':select_speaker_reduced_features(base)};tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['validation_speakers']));Cs=[.3,1.,3.,10.,30.];gfs=[.25,.5,1.,2.]
prepared={}
for name,X in features.items():sc=StandardScaler();prepared[name]=(sc.fit_transform(X[tr]),sc.transform(X[va]))
def one(name,C,gf,balanced):
 Xt,Xv=prepared[name];m=SVC(C=C,gamma=gf/Xt.shape[1],class_weight='balanced' if balanced else None,cache_size=512);m.fit(Xt,y[tr]);p=m.predict(Xv);return {'feature_set':name,'C':C,'gamma_factor':gf,'gamma':gf/Xt.shape[1],'class_weight':'balanced' if balanced else None,'accuracy':float(accuracy_score(y[va],p)),'macro_f1':float(f1_score(y[va],p,average='macro'))}
rows=Parallel(n_jobs=4,prefer='threads')(delayed(one)(n,C,g,b) for n in features for C in Cs for g in gfs for b in (False,True));best=max(rows,key=lambda r:r['macro_f1']);out={'protocol':split['protocol'],'test_status':split['test_status'],'grid':{'feature_set':list(features),'C':Cs,'gamma_factor_over_n_features':gfs,'class_weight':[None,'balanced']},'rows':rows,'frozen_mfcc_config':best};(root/'models/speech/mfcc_clean_v2_frozen_config.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(best,ensure_ascii=False,indent=2))
