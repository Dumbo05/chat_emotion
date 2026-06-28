from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from scipy.signal import resample_poly
from transformers import AutoModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from emotion_app.audio_features import _read_audio
from scripts.train_speech_model import discover_tess,discover_crema,discover_emodb

def args():
 p=argparse.ArgumentParser();p.add_argument('--split-file',type=Path,default=ROOT/'models/speech/wavlm_clean_split_v3.json');p.add_argument('--output-cache',type=Path,default=ROOT/'models/speech/wavlm_clean_v3_train_augmented.npy');p.add_argument('--output-metadata',type=Path,default=ROOT/'models/speech/wavlm_clean_v3_train_augmented_metadata.json');p.add_argument('--augmentation',choices=['none','volume_noise_speed'],default='volume_noise_speed');p.add_argument('--seed',type=int,default=2026);return p.parse_args()
def main():
 a=args();split=json.loads(a.split_file.read_text(encoding='utf8'));assert split['test_status']=='sealed'
 if a.output_cache.exists() or a.output_metadata.exists():raise SystemExit('Refusing to overwrite augmentation cache')
 samples=discover_tess(ROOT/'datasets/TESS')+discover_crema(ROOT/'datasets/CREMA-D')+discover_emodb(ROOT/'datasets/EmoDB');train_set=set(split['train_speakers']);val_test=set(split['val_speakers'])|set(split['test_speakers']);indices=[i for i,s in enumerate(samples) if s.speaker in train_set];assert len(indices)==split['train_sample_count'];assert not any(samples[i].speaker in val_test for i in indices)
 os.environ['HF_HOME']=str((ROOT/'.hf_cache').resolve());model=AutoModel.from_pretrained('microsoft/wavlm-base-plus').cuda().eval().half();[p.requires_grad_(False) for p in model.parameters()];layers=(5,7,9);weights=(.5,.3,.2);out=np.lib.format.open_memmap(a.output_cache,mode='w+',dtype=np.float16,shape=(len(indices),1536));rng=np.random.default_rng(a.seed);batch=12;max_len=64000
 for begin in range(0,len(indices),batch):
  ids=indices[begin:begin+batch];waves=[];lengths=[]
  for idx in ids:
   w=_read_audio(samples[idx].path).astype(np.float32);speed=float(rng.choice([.95,1.05]));up,down=(100,95) if speed==.95 else (100,105);w=resample_poly(w,up,down).astype(np.float32);gain=float(rng.uniform(.85,1.15));w*=gain
   if len(w)>max_len:
    start=int(rng.integers(0,len(w)-max_len+1));w=w[start:start+max_len]
   rms=float(np.sqrt(np.mean(w*w)+1e-9));snr=float(rng.uniform(28,38));noise_rms=rms/(10**(snr/20));w+=rng.normal(0,noise_rms,size=len(w)).astype(np.float32);w=(w-w.mean())/(w.std()+1e-7);waves.append(w);lengths.append(len(w))
  width=max(lengths);x=np.zeros((len(waves),width),np.float32);mask=np.zeros((len(waves),width),np.int64)
  for j,w in enumerate(waves):x[j,:len(w)]=w;mask[j,:len(w)]=1
  tx=torch.from_numpy(x).cuda();tm=torch.from_numpy(mask).cuda()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
   o=model(tx,attention_mask=tm,output_hidden_states=True);fm=model._get_feature_vector_attention_mask(o.last_hidden_state.shape[1],tm).unsqueeze(-1);den=fm.sum(1).clamp_min(1);stats=0
   for layer,weight in zip(layers,weights):
    h=o.hidden_states[layer];mean=(h*fm).sum(1)/den;var=((h-mean[:,None])**2*fm).sum(1)/den;stats=stats+weight*torch.cat([mean,var.clamp_min(1e-7).sqrt()],1)
  out[begin:begin+len(ids)]=stats.float().cpu().numpy().astype(np.float16)
  if (begin//batch+1)%20==0 or begin+len(ids)==len(indices):out.flush();print(f'augment features {begin+len(ids)}/{len(indices)}',flush=True)
 out.flush();meta={'protocol_name':'wavlm_clean_v3','split_file':str(a.split_file.resolve()),'used_test_set':False,'source_partition':'train only','source_indices':indices,'source_speakers_count':len(train_set),'samples':len(indices),'wavlm_model_name':'microsoft/wavlm-base-plus','layers':list(layers),'layer_weights':list(weights),'pooling':'mean_std','augmentation_enabled':True,'augmentation_config':{'volume_gain':[.85,1.15],'gaussian_noise_snr_db':[28,38],'speed':[.95,1.05],'random_crop_max_samples':max_len,'sample_rate':16000},'seed':a.seed,'dtype':'float16','shape':list(out.shape),'created_at':datetime.now(timezone.utc).isoformat()};a.output_metadata.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');print({'cache':str(a.output_cache),'sha256':hashlib.sha256(a.output_cache.read_bytes()).hexdigest()})
if __name__=='__main__':main()
