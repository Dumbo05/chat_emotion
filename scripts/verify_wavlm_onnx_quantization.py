import json,sys,time
from pathlib import Path
import joblib,numpy as np,onnxruntime as ort
sys.path.insert(0,'.')
from emotion_app.audio_features import _read_audio
from scripts.train_simsan import discover_tess,discover_crema,discover_emodb
samples=discover_tess(Path('datasets/TESS'))+discover_crema(Path('datasets/CREMA-D'))+discover_emodb(Path('datasets/EmoDB'))
split=json.loads(Path('models/speech/simsan_split_manifest.json').read_text(encoding='utf8')); ids=[i for i,s in enumerate(samples) if s.speaker in split['validation_speakers']]
rng=np.random.default_rng(20260621); ids=rng.choice(ids,20,replace=False); head=joblib.load('models/speech/wavlm_simsan_head.joblib')['model']; cached=np.load('models/speech/wavlm_layer_stats.npy',mmap_mode='r')
opt=ort.SessionOptions();opt.intra_op_num_threads=4
sf=ort.InferenceSession('models/speech/wavlm_simsan_encoder.onnx',sess_options=opt,providers=['CPUExecutionProvider']);sq=ort.InferenceSession('models/speech/wavlm_simsan_encoder_int8.onnx',sess_options=opt,providers=['CPUExecutionProvider'])
rows=[]
for i in ids:
 w=_read_audio(samples[i].path)[:64000].astype(np.float32);w=(w-w.mean())/(w.std()+1e-7);x=w[None]
 f=sf.run(None,{'input_values':x})[0];q=sq.run(None,{'input_values':x})[0]; c=np.asarray(cached[i,10],np.float32)[None]
 rows.append((int(head.predict(c)[0]),int(head.predict(f)[0]),int(head.predict(q)[0]),float(np.mean(abs(f-q))),float(np.mean(abs(c-f)))))
print({'samples':len(rows),'float_vs_int8_prediction_agreement':sum(a[1]==a[2] for a in rows)/len(rows),'cached_vs_int8_prediction_agreement':sum(a[0]==a[2] for a in rows)/len(rows),'mean_float_int8_abs_diff':float(np.mean([a[3] for a in rows])),'mean_cached_float_abs_diff':float(np.mean([a[4] for a in rows]))})
