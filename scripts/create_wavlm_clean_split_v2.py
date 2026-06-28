from __future__ import annotations
import hashlib, json, random, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,'.')
from scripts.train_speech_model import discover_crema, discover_emodb

ROOT=Path('.'); OUT=ROOT/'models/speech/wavlm_clean_split_v2.json'; OLD=ROOT/'models/speech/simsan_split_manifest.json'; SEED=20260622
if OUT.exists(): raise SystemExit(f'划分已存在，拒绝重建: {OUT}')
samples=discover_crema(ROOT/'datasets/CREMA-D')+discover_emodb(ROOT/'datasets/EmoDB')
old=json.loads(OLD.read_text(encoding='utf8'))
rng=random.Random(SEED)
all_by_dataset={d:sorted({s.speaker for s in samples if s.dataset==d}) for d in ('CREMA-D','EmoDB')}
old_train=set(old['train_speakers'])
parts={'train_speakers':[],'validation_speakers':[],'test_speakers':[]}
counts={'CREMA-D':(18,18),'EmoDB':(2,2)}
for dataset,speakers in all_by_dataset.items():
 test_n,val_n=counts[dataset]
 eligible=[s for s in speakers if s in old_train]
 if len(eligible)<test_n: raise RuntimeError(f'{dataset} 无足够的从未作为测试/验证评估的旧训练说话人')
 test=sorted(rng.sample(eligible,test_n))
 remaining=[s for s in speakers if s not in test]
 validation=sorted(rng.sample(remaining,val_n))
 train=sorted(set(remaining)-set(validation))
 parts['train_speakers']+=train;parts['validation_speakers']+=validation;parts['test_speakers']+=test
fingerprint=hashlib.sha256('\n'.join(s.cache_key for s in samples).encode()).hexdigest()
manifest={'protocol':'WavLM clean speaker-independent evaluation v2','seed':SEED,'created_at_utc':datetime.now(timezone.utc).isoformat(),'datasets':['CREMA-D','EmoDB'],'emotions':['anger','disgust','fear','joy','sadness','neutral'],'exclusion':'TESS excluded because two speakers cannot populate disjoint train/validation/test partitions','dataset_fingerprint':fingerprint,'split_ratio_by_speaker':'approximately 60/20/20, stratified by dataset',**{k:sorted(v) for k,v in parts.items()},'test_provenance':'test speakers sampled only from the previous protocol training speakers; their held-out performance was not inspected before this protocol','selection_policy':{'allowed_data':'train and validation only','hyperparameters':['WavLM layer','SVM C','SVM gamma','class_weight','pooling mean/std/mean+std'],'test_access':'one evaluation after all four configurations are frozen'},'test_status':'sealed-not-evaluated'}
assert not(set(parts['train_speakers'])&set(parts['validation_speakers']) or set(parts['train_speakers'])&set(parts['test_speakers']) or set(parts['validation_speakers'])&set(parts['test_speakers']))
OUT.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'train_speakers':len(parts['train_speakers']),'validation_speakers':len(parts['validation_speakers']),'test_speakers':len(parts['test_speakers']),'test_status':manifest['test_status'],'manifest':str(OUT)},ensure_ascii=False,indent=2))
