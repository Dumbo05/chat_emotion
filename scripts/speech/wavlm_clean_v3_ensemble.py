from __future__ import annotations
import hashlib,itertools,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb
spfile=ROOT/'models/speech/wavlm_clean_split_v3.json';split=json.loads(spfile.read_text(encoding='utf8'));assert split['test_status']=='sealed';samples=discover_tess(ROOT/'datasets/TESS')+discover_crema(ROOT/'datasets/CREMA-D')+discover_emodb(ROOT/'datasets/EmoDB');sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);X=np.load(ROOT/'models/speech/wavlm_clean_v2_layer_stats.npy',mmap_mode='r');tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['val_speakers']));used=[LABELS.index(x) for x in split['emotion_labels']]
members=[{'id':'l7','layers':[7],'weights':[1.],'C':3.,'gf':.5,'cw':None},{'id':'l7b','layers':[7],'weights':[1.],'C':3.,'gf':.5,'cw':'balanced'},{'id':'l57','layers':[5,7],'weights':[.5,.5],'C':3.,'gf':.25,'cw':None},{'id':'l57b','layers':[5,7],'weights':[.5,.5],'C':3.,'gf':.25,'cw':'balanced'},{'id':'l579w','layers':[5,7,9],'weights':[.5,.3,.2],'C':3.,'gf':.25,'cw':None},{'id':'l5b','layers':[5],'weights':[1.],'C':3.,'gf':.5,'cw':'balanced'}]
scores={}
for c in members:
 fused=sum(w*np.asarray(X[:,l,:1536],np.float32) for l,w in zip(c['layers'],c['weights']));sc=StandardScaler();Xt=sc.fit_transform(fused[tr]);Xv=sc.transform(fused[va]);m=SVC(C=c['C'],gamma=c['gf']/Xt.shape[1],class_weight=c['cw'],cache_size=1024,random_state=2026);m.fit(Xt,y[tr]);scores[c['id']]=m.decision_function(Xv)
candidates=[]
for size in range(2,6):
 for ids in itertools.combinations([m['id'] for m in members],size):candidates.append((ids,[1/size]*size))
for ids in itertools.combinations([m['id'] for m in members],2):
 for w in (.25,.75):candidates.append((ids,[w,1-w]))
records=[]
for ids,weights in candidates:
 z=sum(w*scores[i] for i,w in zip(ids,weights));p=np.array([used[i] for i in z.argmax(1)]);rep=classification_report(y[va],p,labels=used,target_names=split['emotion_labels'],output_dict=True,zero_division=0);cfg={'members':ids,'weights':weights};rid=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16];records.append({'run_id':rid,'protocol_name':'wavlm_clean_v3','split_file':str(spfile.resolve()),'used_test_set':False,'dataset_names':split['dataset_names'],'emotion_labels':split['emotion_labels'],'random_seed':2026,'train_speakers_count':len(split['train_speakers']),'val_speakers_count':len(split['val_speakers']),'test_speakers_count':len(split['test_speakers']),'train_sample_count':split['train_sample_count'],'val_sample_count':split['val_sample_count'],'test_sample_count':split['test_sample_count'],'wavlm_model_name':'microsoft/wavlm-base-plus','wavlm_layers':sorted(set(sum((next(m['layers'] for m in members if m['id']==i) for i in ids),[]))),'layer_fusion':'decision_score_ensemble','ensemble_members':list(ids),'ensemble_weights':weights,'pooling':'mean_std','feature_dim':1536,'classifier':'rbf_svm_ensemble','C':None,'gamma':None,'class_weight':'mixed','scaler_type':'Per-member StandardScaler fitted on train only','augmentation_enabled':False,'augmentation_config':{'name':'none'},'val_accuracy':float(accuracy_score(y[va],p)),'val_macro_f1':float(f1_score(y[va],p,average='macro')),'val_weighted_f1':float(f1_score(y[va],p,average='weighted')),'val_per_class_precision':{x:float(rep[x]['precision']) for x in split['emotion_labels']},'val_per_class_recall':{x:float(rep[x]['recall']) for x in split['emotion_labels']},'val_per_class_f1':{x:float(rep[x]['f1-score']) for x in split['emotion_labels']},'val_confusion_matrix':confusion_matrix(y[va],p,labels=used).tolist(),'model_save_path':None,'created_at':datetime.now(timezone.utc).isoformat(),'notes':'Validation-only RBF-SVM decision-score ensemble; v3 test sealed. Member definitions stored in ensemble summary.'})
log=ROOT/'outputs/speech/wavlm_clean_v3_validation_sweep.jsonl'
with log.open('a',encoding='utf8') as f:
 for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
best=max(records,key=lambda r:(r['val_macro_f1'],r['val_accuracy']));summary={'protocol_name':'wavlm_clean_v3','used_test_set':False,'test_status':split['test_status'],'member_definitions':members,'candidate_count':len(records),'best':best,'created_at':datetime.now(timezone.utc).isoformat()};(ROOT/'models/speech/wavlm_clean_v3_ensemble_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:best[k] for k in ('run_id','ensemble_members','ensemble_weights','val_accuracy','val_macro_f1')},ensure_ascii=False,indent=2))
