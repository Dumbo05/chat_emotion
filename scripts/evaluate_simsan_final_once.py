import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
sys.path.insert(0,'.')
from scripts.train_simsan import discover_tess, discover_crema, discover_emodb, LABELS, LABEL_TO_ID, SpectrogramDataset
from emotion_app.simsan import SIMSAN

out=Path('models/speech/simsan_final_test_metrics.json')
if out.exists(): raise SystemExit('拒绝重复评估：最终测试结果文件已存在')
samples=discover_tess(Path('datasets/TESS'))+discover_crema(Path('datasets/CREMA-D'))+discover_emodb(Path('datasets/EmoDB'))
split_path=Path('models/speech/simsan_split_manifest.json')
split=json.loads(split_path.read_text(encoding='utf-8'))
sp=np.array([s.speaker for s in samples]); y=np.array([LABEL_TO_ID[s.label] for s in samples])
idx=np.flatnonzero(np.isin(sp,split['final_test_speakers']))
assert len(idx)>0 and not np.isin(sp[idx],split['train_speakers']+split['validation_speakers']).any()
ds=SpectrogramDataset(Path('models/speech/simsan_logmel.npy'),idx,y,np.full(len(y),-1),'none')
loader=DataLoader(ds,batch_size=128,shuffle=False)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(path):
 ck=torch.load(path,map_location='cpu',weights_only=True)
 m=SIMSAN(len(LABELS),ck['speaker_classes']);m.load_state_dict(ck['state_dict']);m.to(device).eval(); out=[]
 with torch.inference_mode():
  for x,_,_ in loader:
   z,_=m(x.to(device),0.0);out.append(z.cpu().numpy())
 return np.concatenate(out)

z=.76*predict('models/speech/simsan_best.pt')+.24*predict('models/speech/simsan_balanced.pt')
yt=y[idx]; yp=z.argmax(1); used=sorted(np.unique(yt).tolist())
report={
 'architecture':'SIMSAN-v1 dual-checkpoint ensemble',
 'evaluation_policy':'locked speaker-independent final test; evaluated once after configuration freeze',
 'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
 'split_manifest_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
 'ensemble_weights':{'simsan_best.pt':.76,'simsan_balanced.pt':.24},
 'test_speakers':split['final_test_speakers'],'samples':int(len(yt)),
 'labels_evaluated':[LABELS[i] for i in used],
 'accuracy':float(accuracy_score(yt,yp)),
 'precision_macro':float(precision_score(yt,yp,labels=used,average='macro',zero_division=0)),
 'recall_macro':float(recall_score(yt,yp,labels=used,average='macro',zero_division=0)),
 'f1_macro':float(f1_score(yt,yp,labels=used,average='macro',zero_division=0)),
 'classification_report':classification_report(yt,yp,labels=used,target_names=[LABELS[i] for i in used],output_dict=True,zero_division=0),
 'confusion_matrix':confusion_matrix(yt,yp,labels=used).tolist(),
 'note':'Surprise has no untouched speaker in the supplied datasets and is therefore excluded from final test; it remains covered by validation.'
}
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['samples','labels_evaluated','accuracy','precision_macro','recall_macro','f1_macro']},ensure_ascii=False,indent=2))
