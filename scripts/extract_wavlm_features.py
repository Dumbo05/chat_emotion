from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
sys.path.insert(0,'.')
from emotion_app.audio_features import _read_audio
from scripts.train_speech_model import discover_tess, discover_crema, discover_emodb

ROOT=Path('.'); OUT=ROOT/'models/speech/wavlm_layer_stats.npy'; META=ROOT/'models/speech/wavlm_layer_stats_metadata.json'
samples=discover_tess(ROOT/'datasets/TESS')+discover_crema(ROOT/'datasets/CREMA-D')+discover_emodb(ROOT/'datasets/EmoDB')
fingerprint=hashlib.sha256('\n'.join(s.cache_key for s in samples).encode()).hexdigest()
if OUT.exists() and META.exists() and json.loads(META.read_text(encoding='utf8')).get('fingerprint')==fingerprint:
 print('复用',OUT); raise SystemExit
model=AutoModel.from_pretrained('microsoft/wavlm-base-plus').cuda().eval().half()
for p in model.parameters(): p.requires_grad_(False)
layers=model.config.num_hidden_layers+1; dim=model.config.hidden_size
arr=np.lib.format.open_memmap(OUT,mode='w+',dtype=np.float16,shape=(len(samples),layers,dim*2))
batch_size=12; max_len=64000
for begin in range(0,len(samples),batch_size):
 chunk=samples[begin:begin+batch_size]; waves=[]; lengths=[]
 for s in chunk:
  w=_read_audio(s.path)[:max_len].astype(np.float32)
  w=(w-w.mean())/(w.std()+1e-7); lengths.append(len(w)); waves.append(w)
 width=max(lengths); x=np.zeros((len(waves),width),np.float32); mask=np.zeros((len(waves),width),np.int64)
 for i,w in enumerate(waves): x[i,:len(w)]=w; mask[i,:len(w)]=1
 with torch.inference_mode(), torch.amp.autocast('cuda',dtype=torch.float16):
  o=model(torch.from_numpy(x).cuda(),attention_mask=torch.from_numpy(mask).cuda(),output_hidden_states=True)
  feat_mask=model._get_feature_vector_attention_mask(o.last_hidden_state.shape[1],torch.from_numpy(mask).cuda()).unsqueeze(-1)
  denom=feat_mask.sum(1).clamp_min(1)
  for li,h in enumerate(o.hidden_states):
   mean=(h*feat_mask).sum(1)/denom
   var=((h-mean[:,None])**2*feat_mask).sum(1)/denom
   stats=torch.cat([mean,var.clamp_min(1e-7).sqrt()],1)
   arr[begin:begin+len(chunk),li]=stats.float().cpu().numpy().astype(np.float16)
 if (begin//batch_size+1)%20==0 or begin+len(chunk)==len(samples):
  arr.flush(); print(f'WavLM {begin+len(chunk)}/{len(samples)}',flush=True)
arr.flush(); META.write_text(json.dumps({'fingerprint':fingerprint,'model':'microsoft/wavlm-base-plus','shape':list(arr.shape),'dtype':'float16','pooling':'masked mean+std per hidden layer'},ensure_ascii=False,indent=2),encoding='utf8')
