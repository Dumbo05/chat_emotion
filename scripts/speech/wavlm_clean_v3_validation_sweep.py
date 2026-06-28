from __future__ import annotations
import argparse,hashlib,json,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from joblib import Parallel,delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb

def csv_values(text,cast=str):return [cast(x.strip()) for x in text.split(',') if x.strip()]
def layer_groups(text):return [tuple(int(x) for x in group.split('+')) for group in text.split(';') if group.strip()]
def parse_args():
 p=argparse.ArgumentParser(description='Validation-only WavLM clean v3 sweep')
 p.add_argument('--split-file',type=Path,default=PROJECT_ROOT/'models/speech/wavlm_clean_split_v3.json')
 p.add_argument('--output-jsonl',type=Path,default=PROJECT_ROOT/'outputs/speech/wavlm_clean_v3_validation_sweep.jsonl')
 p.add_argument('--output-summary',type=Path,default=PROJECT_ROOT/'models/speech/wavlm_clean_v3_validation_summary.json')
 p.add_argument('--feature-cache',type=Path,default=PROJECT_ROOT/'models/speech/wavlm_clean_v2_layer_stats.npy')
 p.add_argument('--layers',default='5,7,9,11')
 p.add_argument('--layer-groups',default='5+7;7+9;5+7+9;7+9+11;5+7+9+11')
 p.add_argument('--pooling',default='mean,mean_std')
 p.add_argument('--classifiers',default='rbf_svm,logistic_regression,linear_svc')
 p.add_argument('--C-values',default='1,3,5,10')
 p.add_argument('--logistic-C-values',default='0.3,1,3,10')
 p.add_argument('--linear-C-values',default='0.01,0.03,0.1,0.3,1')
 p.add_argument('--gamma-values',default='scale,0.25,0.5,1.0')
 p.add_argument('--class-weight-options',default='none,balanced')
 p.add_argument('--augmentation',default='none',choices=['none','volume_noise_speed'])
 p.add_argument('--seed',type=int,default=2026)
 p.add_argument('--jobs',type=int,default=4)
 p.add_argument('--append',action='store_true')
 p.add_argument('--save-group-winners',action='store_true')
 p.add_argument('--notes',default='')
 return p.parse_args()

def main():
 a=parse_args();split=json.loads(a.split_file.read_text(encoding='utf8'))
 if split['protocol_name']!='wavlm_clean_v3' or split['test_status']!='sealed':raise RuntimeError('v3 test must remain sealed during validation sweep')
 if a.augmentation!='none':raise NotImplementedError('Waveform augmentation is intentionally disabled until a train-only cache is implemented and audited.')
 if a.output_jsonl.exists() and not a.append:raise SystemExit(f'Refusing to overwrite sweep log; use --append: {a.output_jsonl}')
 samples=discover_tess(PROJECT_ROOT/'datasets/TESS')+discover_crema(PROJECT_ROOT/'datasets/CREMA-D')+discover_emodb(PROJECT_ROOT/'datasets/EmoDB')
 sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);X=np.load(a.feature_cache,mmap_mode='r')
 mf=np.load(PROJECT_ROOT/'models/speech/multidataset_features.npz');assert len(X)==len(samples) and np.array_equal(sp,mf['speakers'])
 tr=np.flatnonzero(np.isin(sp,split['train_speakers']));va=np.flatnonzero(np.isin(sp,split['val_speakers']))
 assert len(tr)==split['train_sample_count'] and len(va)==split['val_sample_count']
 # Deliberately never construct a test index or read test targets here.
 singles=[(x,) for x in csv_values(a.layers,int)];groups=layer_groups(a.layer_groups);features=[]
 for group in singles+groups:
  if group not in features:features.append(group)
 pools=csv_values(a.pooling);classifiers=csv_values(a.classifiers);weights=[None if x.lower()=='none' else 'balanced' for x in csv_values(a.class_weight_options)]
 rbf_C=csv_values(a.C_values,float);lr_C=csv_values(a.logistic_C_values,float);lin_C=csv_values(a.linear_C_values,float);gammas=csv_values(a.gamma_values)
 used=[LABELS.index(x) for x in split['emotion_labels']];records=[];a.output_jsonl.parent.mkdir(parents=True,exist_ok=True)
 for layers in features:
  fused=np.asarray(X[:,layers,:],np.float32).mean(axis=1)
  for pool in pools:
   raw=fused[:,:768] if pool=='mean' else fused[:,:1536]
   scaler=StandardScaler();Xt=scaler.fit_transform(raw[tr]);Xv=scaler.transform(raw[va]);dim=Xt.shape[1]
   configs=[]
   if 'rbf_svm' in classifiers:
    for C in rbf_C:
     for g in gammas:
      for weight in weights:configs.append(('rbf_svm',C,g,weight))
   if 'logistic_regression' in classifiers:
    for C in lr_C:
     for weight in weights:configs.append(('logistic_regression',C,None,weight))
   if 'linear_svc' in classifiers:
    for C in lin_C:
     for weight in weights:configs.append(('linear_svc',C,None,weight))
   def evaluate(config):
    classifier,C,gamma_value,weight=config
    if classifier=='rbf_svm':
     gamma='scale' if gamma_value=='scale' else float(gamma_value)/dim
     model=SVC(C=C,gamma=gamma,class_weight=weight,cache_size=1024,random_state=a.seed)
    elif classifier=='logistic_regression':model=LogisticRegression(C=C,class_weight=weight,max_iter=3000,random_state=a.seed)
    else:model=LinearSVC(C=C,class_weight=weight,max_iter=5000,dual='auto',random_state=a.seed)
    model.fit(Xt,y[tr]);pred=model.predict(Xv);rep=classification_report(y[va],pred,labels=used,target_names=split['emotion_labels'],output_dict=True,zero_division=0)
    config_doc={'layers':list(layers),'pooling':pool,'classifier':classifier,'C':C,'gamma':gamma_value,'class_weight':weight}
    run_id=hashlib.sha256(json.dumps(config_doc,sort_keys=True).encode()).hexdigest()[:16]
    return {'run_id':run_id,'protocol_name':'wavlm_clean_v3','split_file':str(a.split_file.resolve()),'used_test_set':False,'dataset_names':split['dataset_names'],'emotion_labels':split['emotion_labels'],'random_seed':a.seed,'train_speakers_count':len(split['train_speakers']),'val_speakers_count':len(split['val_speakers']),'test_speakers_count':len(split['test_speakers']),'train_sample_count':split['train_sample_count'],'val_sample_count':split['val_sample_count'],'test_sample_count':split['test_sample_count'],'wavlm_model_name':'microsoft/wavlm-base-plus','wavlm_layers':list(layers),'layer_fusion':'arithmetic_mean','pooling':pool,'feature_dim':dim,'classifier':classifier,'C':C,'gamma':gamma_value,'class_weight':weight,'scaler_type':'StandardScaler fitted on train only','augmentation_enabled':False,'augmentation_config':{'name':'none'},'val_accuracy':float(accuracy_score(y[va],pred)),'val_macro_f1':float(f1_score(y[va],pred,average='macro')),'val_weighted_f1':float(f1_score(y[va],pred,average='weighted')),'val_per_class_precision':{x:float(rep[x]['precision']) for x in split['emotion_labels']},'val_per_class_recall':{x:float(rep[x]['recall']) for x in split['emotion_labels']},'val_per_class_f1':{x:float(rep[x]['f1-score']) for x in split['emotion_labels']},'val_confusion_matrix':confusion_matrix(y[va],pred,labels=used).tolist(),'model_save_path':None,'created_at':datetime.now(timezone.utc).isoformat(),'notes':a.notes or 'Validation-only candidate; v3 test index and labels were not accessed.'},model,scaler
   evaluated=Parallel(n_jobs=a.jobs,prefer='threads')(delayed(evaluate)(c) for c in configs)
   group_records=[x[0] for x in evaluated];winner_index=max(range(len(group_records)),key=lambda i:(group_records[i]['val_macro_f1'],group_records[i]['val_accuracy']))
   if a.save_group_winners:
    rec,model,sc=evaluated[winner_index];path=PROJECT_ROOT/'models/speech/clean_v3_validation_candidates'/f'{rec["run_id"]}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':model,'scaler':sc,'record':rec},path,compress=3);rec['model_save_path']=str(path.resolve())
   with a.output_jsonl.open('a',encoding='utf8') as f:
    for rec,_,_ in evaluated:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
   records.extend(group_records);best=max(group_records,key=lambda r:r['val_macro_f1']);print({'layers':layers,'pooling':pool,'best_f1':round(best['val_macro_f1'],4),'classifier':best['classifier'],'C':best['C'],'gamma':best['gamma'],'weight':best['class_weight']},flush=True)
 all_records=[]
 with a.output_jsonl.open(encoding='utf8') as f:
  for line in f:
   if line.strip():all_records.append(json.loads(line))
 # Deduplicate by run_id while keeping the latest record.
 unique={r['run_id']:r for r in all_records};ranked=sorted(unique.values(),key=lambda r:(r['val_macro_f1'],r['val_accuracy']),reverse=True)
 summary={'protocol_name':'wavlm_clean_v3','split_file':str(a.split_file.resolve()),'used_test_set':False,'test_status':split['test_status'],'candidate_count':len(ranked),'feature_cache':str(a.feature_cache.resolve()),'feature_cache_sha256':hashlib.sha256(a.feature_cache.read_bytes()).hexdigest(),'best_validation_config':ranked[0],'top_10':ranked[:10],'created_at':datetime.now(timezone.utc).isoformat()};a.output_summary.parent.mkdir(parents=True,exist_ok=True);a.output_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print('BEST',json.dumps({k:ranked[0][k] for k in ('run_id','wavlm_layers','pooling','classifier','C','gamma','class_weight','val_accuracy','val_macro_f1')},ensure_ascii=False))
if __name__=='__main__':main()
