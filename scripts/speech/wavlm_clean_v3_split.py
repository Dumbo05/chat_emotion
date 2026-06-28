from __future__ import annotations
import argparse,json,random,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from scripts.train_speech_model import discover_crema,discover_emodb

def parse_args():
 p=argparse.ArgumentParser(description='Create sealed WavLM clean v3 speaker split')
 p.add_argument('--split-file',type=Path,default=PROJECT_ROOT/'models/speech/wavlm_clean_split_v3.json')
 p.add_argument('--v2-split-file',type=Path,default=PROJECT_ROOT/'models/speech/wavlm_clean_split_v2.json')
 p.add_argument('--crema-root',type=Path,default=PROJECT_ROOT/'datasets/CREMA-D')
 p.add_argument('--emodb-root',type=Path,default=PROJECT_ROOT/'datasets/EmoDB')
 p.add_argument('--seed',type=int,default=2026)
 return p.parse_args()

def main():
 a=parse_args()
 if a.split_file.exists():raise SystemExit(f'Refusing to overwrite existing split: {a.split_file}')
 if not a.v2_split_file.is_file():raise FileNotFoundError(a.v2_split_file)
 samples=discover_crema(a.crema_root)+discover_emodb(a.emodb_root)
 v2=json.loads(a.v2_split_file.read_text(encoding='utf8'))
 v2_test=set(v2['test_speakers']);v2_train=set(v2['train_speakers'])
 by_dataset={d:sorted({s.speaker for s in samples if s.dataset==d and s.speaker not in v2_test}) for d in ('CREMA-D','EmoDB')}
 target={'CREMA-D':{'test':15,'val':15},'EmoDB':{'test':2,'val':2}}
 rng=random.Random(a.seed);parts={'train':[],'val':[],'test':[]}
 for dataset,speakers in by_dataset.items():
  eligible=sorted(set(speakers)&v2_train)
  test=sorted(rng.sample(eligible,target[dataset]['test']))
  remain=sorted(set(speakers)-set(test))
  val=sorted(rng.sample(remain,target[dataset]['val']))
  train=sorted(set(remain)-set(val))
  parts['train']+=train;parts['val']+=val;parts['test']+=test
 sets={k:set(v) for k,v in parts.items()}
 assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
 assert not((sets['train']|sets['val']|sets['test'])&v2_test)
 emotion_labels=['anger','disgust','fear','joy','sadness','neutral']
 selected=[s for s in samples if s.speaker not in v2_test]
 def metadata(key):
  rows=[s for s in selected if s.speaker in sets[key]]
  dist=Counter(s.label for s in rows)
  if set(dist)!=set(emotion_labels):raise RuntimeError(f'{key} does not cover all emotions: {dist}')
  return len(rows),{label:int(dist[label]) for label in emotion_labels}
 train_n,train_d=metadata('train');val_n,val_d=metadata('val');test_n,test_d=metadata('test')
 doc={'protocol_name':'wavlm_clean_v3','dataset_names':['CREMA-D','EmoDB'],'excluded_datasets':{'TESS':'Only two speakers; cannot form mutually exclusive train/validation/test partitions.','wavlm_clean_v2_test_speakers':'All v2 test speakers excluded from every v3 partition.'},'emotion_labels':emotion_labels,'random_seed':a.seed,'train_speakers':sorted(parts['train']),'val_speakers':sorted(parts['val']),'test_speakers':sorted(parts['test']),'train_sample_count':train_n,'val_sample_count':val_n,'test_sample_count':test_n,'per_split_class_distribution':{'train':train_d,'validation':val_d,'test':test_d},'speaker_counts':{k:len(v) for k,v in parts.items()},'test_status':'sealed','created_at':datetime.now(timezone.utc).isoformat(),'v2_test_exclusion_count':len(v2_test),'test_provenance':'v3 test speakers sampled only from v2 train speakers; they were not v2 validation/test speakers.','statement':'Train, validation, and test speakers are mutually exclusive. No v2 test speaker is used by v3.'}
 a.split_file.parent.mkdir(parents=True,exist_ok=True);a.split_file.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'split_file':str(a.split_file),'speaker_counts':doc['speaker_counts'],'sample_counts':{'train':train_n,'val':val_n,'test':test_n},'test_status':doc['test_status'],'v2_test_speakers_excluded':len(v2_test)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
