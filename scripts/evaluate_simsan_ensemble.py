import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, '.')
from scripts.train_simsan import discover_tess, discover_crema, discover_emodb, LABELS, LABEL_TO_ID, SpectrogramDataset
from emotion_app.simsan import SIMSAN

samples = discover_tess(Path('datasets/TESS')) + discover_crema(Path('datasets/CREMA-D')) + discover_emodb(Path('datasets/EmoDB'))
split = json.loads(Path('models/speech/simsan_split_manifest.json').read_text(encoding='utf-8'))
speakers = np.array([s.speaker for s in samples])
y = np.array([LABEL_TO_ID[s.label] for s in samples])
idx = np.flatnonzero(np.isin(speakers, split['validation_speakers']))
ds = SpectrogramDataset(Path('models/speech/simsan_logmel.npy'), idx, y, np.full(len(y), -1), 'none')
loader = DataLoader(ds, batch_size=128, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = SIMSAN(len(LABELS), ck['speaker_classes'])
    m.load_state_dict(ck['state_dict']); m.to(device).eval()
    out=[]
    with torch.inference_mode():
        for x,_,_ in loader:
            z,_=m(x.to(device),0.0); out.append(z.cpu().numpy())
    return np.concatenate(out)

paths=['models/speech/simsan_best.pt','models/speech/simsan_balanced.pt','models/speech/simsan_mild.pt']
logits=[predict(p) for p in paths]
yv=y[idx]
for p,z in zip(paths,logits):
    pr=z.argmax(1); print(p, accuracy_score(yv,pr),f1_score(yv,pr,average='macro'))
print('search ensembles')
best=(0,None)
for t0 in np.linspace(.5,2.0,16):
 for t1 in np.linspace(.5,2.0,16):
  for a in np.linspace(0,1,41):
   z=a*logits[0]/t0+(1-a)*logits[1]/t1
   pr=z.argmax(1); score=f1_score(yv,pr,average='macro')
   if score>best[0]: best=(score,(a,t0,t1,accuracy_score(yv,pr)))
print('BEST',best)
np.savez_compressed('models/speech/simsan_validation_logits.npz', y=yv, indices=idx, best=logits[0], balanced=logits[1], mild=logits[2])
