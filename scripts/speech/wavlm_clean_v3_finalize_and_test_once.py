from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.train_speech_model import LABELS,discover_tess,discover_crema,discover_emodb

def args():
 p=argparse.ArgumentParser(description='Freeze v3 config, then evaluate sealed test exactly once')
 p.add_argument('--mode',choices=['freeze','test'],required=True);p.add_argument('--split-file',type=Path,default=ROOT/'models/speech/wavlm_clean_split_v3.json');p.add_argument('--output-jsonl',type=Path,default=ROOT/'outputs/speech/wavlm_clean_v3_validation_sweep.jsonl');p.add_argument('--output-summary',type=Path,default=ROOT/'models/speech/wavlm_clean_v3_validation_summary.json');p.add_argument('--frozen-plan',type=Path,default=ROOT/'models/speech/wavlm_clean_v3_frozen_evaluation_plan.json');p.add_argument('--final-results',type=Path,default=ROOT/'models/speech/wavlm_clean_v3_final_results.json');p.add_argument('--feature-cache',type=Path,default=ROOT/'models/speech/wavlm_clean_v2_layer_stats.npy');p.add_argument('--seed',type=int,default=2026);return p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_records(path):
 records={}
 with path.open(encoding='utf8') as f:
  for line in f:
   if line.strip():
    r=json.loads(line);assert r['used_test_set'] is False;records[r['run_id']]=r
 return list(records.values())
def freeze(a):
 if a.frozen_plan.exists():raise SystemExit(f'Refusing to overwrite frozen plan: {a.frozen_plan}')
 split=json.loads(a.split_file.read_text(encoding='utf8'));assert split['test_status']=='sealed';records=load_records(a.output_jsonl);ranked=sorted(records,key=lambda r:(r['val_macro_f1'],r['val_accuracy'],-len(r.get('wavlm_layers',[]))),reverse=True);selected=ranked[0]
 ensemble=json.loads((ROOT/'models/speech/wavlm_clean_v3_ensemble_refined_summary.json').read_text(encoding='utf8'));member_defs={m['id']:m for m in ensemble['member_definitions']};selected_members={i:member_defs[i] for i in selected.get('ensemble_members',[])}
 summary={'protocol_name':'wavlm_clean_v3','split_file':str(a.split_file.resolve()),'used_test_set':False,'test_status':split['test_status'],'candidate_count':len(ranked),'selection_metric':'val_macro_f1','best_validation_config':selected,'top_10':ranked[:10],'created_at':datetime.now(timezone.utc).isoformat()};a.output_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 plan={'protocol_name':'wavlm_clean_v3','selected_run_id':selected['run_id'],'selected_config':selected,'ensemble_member_definitions':selected_members,'selection_metric':'val_macro_f1','validation_results':{'accuracy':selected['val_accuracy'],'macro_f1':selected['val_macro_f1'],'weighted_f1':selected['val_weighted_f1'],'per_class_precision':selected['val_per_class_precision'],'per_class_recall':selected['val_per_class_recall'],'per_class_f1':selected['val_per_class_f1'],'confusion_matrix':selected['val_confusion_matrix']},'split_file':str(a.split_file.resolve()),'split_file_sha256_before_test':sha(a.split_file),'validation_sweep_file':str(a.output_jsonl.resolve()),'validation_sweep_sha256':sha(a.output_jsonl),'feature_cache':str(a.feature_cache.resolve()),'feature_cache_sha256':sha(a.feature_cache),'test_status_before_final_eval':'sealed','statement':'No test-set metrics were used for model selection.','final_refit_policy':'Refit frozen ensemble members on v3 train + validation speakers; evaluate v3 test once.','created_at':datetime.now(timezone.utc).isoformat()};a.frozen_plan.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'frozen_plan':str(a.frozen_plan),'selected_run_id':selected['run_id'],'val_accuracy':selected['val_accuracy'],'val_macro_f1':selected['val_macro_f1'],'test_status':split['test_status']},ensure_ascii=False,indent=2))
def test_once(a):
 if a.final_results.exists():raise SystemExit('Refusing repeated v3 test: final results already exist')
 if not a.frozen_plan.is_file():raise FileNotFoundError('Frozen plan must exist before test')
 split=json.loads(a.split_file.read_text(encoding='utf8'));plan=json.loads(a.frozen_plan.read_text(encoding='utf8'));assert split['test_status']=='sealed' and plan['test_status_before_final_eval']=='sealed';assert sha(a.split_file)==plan['split_file_sha256_before_test'];assert plan['statement']=='No test-set metrics were used for model selection.'
 samples=discover_tess(ROOT/'datasets/TESS')+discover_crema(ROOT/'datasets/CREMA-D')+discover_emodb(ROOT/'datasets/EmoDB');sp=np.array([s.speaker for s in samples]);y=np.array([LABELS.index(s.label) for s in samples]);X=np.load(a.feature_cache,mmap_mode='r');mf=np.load(ROOT/'models/speech/multidataset_features.npz');assert len(X)==len(samples) and np.array_equal(sp,mf['speakers']);fit=np.flatnonzero(np.isin(sp,split['train_speakers']+split['val_speakers']));te=np.flatnonzero(np.isin(sp,split['test_speakers']));assert not(set(sp[fit])&set(sp[te]));assert len(te)==split['test_sample_count']
 members=[];classes=None
 for member_id in plan['selected_config']['ensemble_members']:
  c=plan['ensemble_member_definitions'][member_id];fused=sum(float(w)*np.asarray(X[:,int(l),:1536],np.float32) for l,w in zip(c['layers'],c['weights']));sc=StandardScaler();Xt=sc.fit_transform(fused[fit]);Xte=sc.transform(fused[te]);m=SVC(C=float(c['C']),gamma=float(c['gf'])/Xt.shape[1],class_weight=c['cw'],cache_size=2048,random_state=a.seed);m.fit(Xt,y[fit]);
  if classes is None:classes=m.classes_
  else:assert np.array_equal(classes,m.classes_)
  members.append({'id':member_id,'config':c,'scaler':sc,'model':m,'test_scores':m.decision_function(Xte)})
 weights=plan['selected_config']['ensemble_weights'];scores=sum(float(w)*m['test_scores'] for w,m in zip(weights,members));pred=classes[scores.argmax(1)];used=[LABELS.index(x) for x in split['emotion_labels']];rep=classification_report(y[te],pred,labels=used,target_names=split['emotion_labels'],output_dict=True,zero_division=0)
 model_dir=ROOT/'models/speech/clean_v3_models';model_dir.mkdir(parents=True,exist_ok=True);model_path=model_dir/'wavlm_clean_v3_best.joblib';bundle_members=[{k:v for k,v in m.items() if k!='test_scores'} for m in members];joblib.dump({'protocol_name':'wavlm_clean_v3','members':bundle_members,'ensemble_weights':weights,'classes':classes,'labels':LABELS,'selected_run_id':plan['selected_run_id'],'feature_cache_hash':plan['feature_cache_sha256']},model_path,compress=3)
 result={'protocol_name':'wavlm_clean_v3','split_file':str(a.split_file.resolve()),'frozen_plan_file':str(a.frozen_plan.resolve()),'selected_config':plan['selected_config'],'test_status':'evaluated-once','test_accuracy':float(accuracy_score(y[te],pred)),'test_macro_f1':float(f1_score(y[te],pred,average='macro')),'test_weighted_f1':float(f1_score(y[te],pred,average='weighted')),'test_per_class_precision':{x:float(rep[x]['precision']) for x in split['emotion_labels']},'test_per_class_recall':{x:float(rep[x]['recall']) for x in split['emotion_labels']},'test_per_class_f1':{x:float(rep[x]['f1-score']) for x in split['emotion_labels']},'test_per_class_support':{x:int(rep[x]['support']) for x in split['emotion_labels']},'test_confusion_matrix':confusion_matrix(y[te],pred,labels=used).tolist(),'test_sample_count':int(len(te)),'test_speakers_count':len(split['test_speakers']),'final_fit_sample_count':int(len(fit)),'final_fit_speakers_count':len(split['train_speakers'])+len(split['val_speakers']),'model_path':str(model_path.resolve()),'model_hash':sha(model_path),'feature_cache_hash':plan['feature_cache_sha256'],'frozen_plan_hash':sha(a.frozen_plan),'created_at':datetime.now(timezone.utc).isoformat(),'statement':'v3 test was evaluated once after configuration freeze. No v2 test speaker was used.'};a.final_results.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');split['test_status']='evaluated-once';split['test_evaluated_at']=result['created_at'];split['final_results_file']=str(a.final_results.resolve());a.split_file.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:result[k] for k in ('test_status','test_accuracy','test_macro_f1','test_weighted_f1','test_sample_count','test_speakers_count','model_hash')},ensure_ascii=False,indent=2))
def main():
 a=args();freeze(a) if a.mode=='freeze' else test_once(a)
if __name__=='__main__':main()
