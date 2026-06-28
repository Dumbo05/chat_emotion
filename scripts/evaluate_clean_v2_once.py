from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
sys.path.insert(0,'.')
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb
root=Path('.');model_root=root/'models/speech';result_path=model_root/'wavlm_clean_v2_final_results.json';split_path=model_root/'wavlm_clean_split_v2.json';plan_path=model_root/'wavlm_clean_v2_frozen_evaluation_plan.json'
if result_path.exists():raise SystemExit('拒绝重复评估：最终结果已存在')
split=json.loads(split_path.read_text(encoding='utf8'));plan=json.loads(plan_path.read_text(encoding='utf8'));assert split['test_status']=='sealed-not-evaluated';assert hashlib.sha256(split_path.read_bytes()).hexdigest()==plan['split_manifest_sha256']
samples=discover_tess(root/'datasets/TESS')+discover_crema(root/'datasets/CREMA-D')+discover_emodb(root/'datasets/EmoDB');sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);wx=np.load(model_root/'wavlm_clean_v2_layer_stats.npy',mmap_mode='r');mf=np.load(model_root/'multidataset_features.npz');assert len(samples)==len(wx)==len(mf['X']);assert np.array_equal(sp,mf['speakers']);assert np.array_equal(np.array([s.label for s in samples]),mf['y'])
fit_speakers=split['train_speakers']+split['validation_speakers'];fit_idx=np.flatnonzero(np.isin(sp,fit_speakers));test_idx=np.flatnonzero(np.isin(sp,split['test_speakers']));assert not(set(sp[fit_idx])&set(sp[test_idx]));used=[LABELS.index(x) for x in split['emotions']];out_dir=model_root/'clean_v2_models';out_dir.mkdir(exist_ok=True);results=[]
for item in plan['models']:
 c=item['config'];mid=item['id']
 if mid=='mfcc_rbf_svm':X=mf['X'];layer=None;pooling='full_177'
 else:
  layer=int(c['layer']);pooling=c['pooling'];sl={'mean':slice(0,768),'std':slice(768,1536),'mean_std':slice(0,1536)}[pooling];X=wx[:,layer,sl]
 classifier=SVC(C=float(c['C']),gamma=float(c['gamma']),kernel='rbf',class_weight=c['class_weight'],cache_size=2048)
 pipeline=make_pipeline(StandardScaler(),classifier);pipeline.fit(np.asarray(X[fit_idx],np.float32),y[fit_idx]);pred=pipeline.predict(np.asarray(X[test_idx],np.float32));report=classification_report(y[test_idx],pred,labels=used,target_names=[LABELS[i] for i in used],output_dict=True,zero_division=0)
 model_path=out_dir/f'{mid}.joblib';joblib.dump({'model':pipeline,'labels':LABELS,'layer':layer,'pooling':pooling,'protocol':split['protocol'],'config':c},model_path,compress=3)
 metrics={'id':mid,'display':item['display'],'feature':item['feature'],'classifier':item['classifier'],'config':c,'validation_accuracy':c['accuracy'],'validation_macro_f1':c['macro_f1'],'test_samples':int(len(test_idx)),'test_accuracy':float(accuracy_score(y[test_idx],pred)),'test_precision_macro':float(precision_score(y[test_idx],pred,labels=used,average='macro',zero_division=0)),'test_recall_macro':float(recall_score(y[test_idx],pred,labels=used,average='macro',zero_division=0)),'test_macro_f1':float(f1_score(y[test_idx],pred,labels=used,average='macro',zero_division=0)),'classification_report':report,'confusion_matrix':confusion_matrix(y[test_idx],pred,labels=used).tolist(),'model_path':str(model_path),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()};results.append(metrics);print(mid,metrics['test_accuracy'],metrics['test_macro_f1'],flush=True)
final={'protocol':split['protocol'],'evidence_level':'pre-registered fresh speaker-disjoint test; evaluated once','evaluated_at_utc':datetime.now(timezone.utc).isoformat(),'split_manifest_sha256_before_status_update':plan['split_manifest_sha256'],'fit_speakers':len(set(sp[fit_idx])),'test_speakers':len(set(sp[test_idx])),'fit_samples':int(len(fit_idx)),'test_samples':int(len(test_idx)),'models':results}
result_path.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8');split['test_status']='evaluated-once';split['test_evaluated_at_utc']=final['evaluated_at_utc'];split['final_results_file']=str(result_path);split_path.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf8');print('FINAL_WRITTEN',result_path)
