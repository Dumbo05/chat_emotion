import json,numpy as np,torch,sys
sys.path.insert(0,".")
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader
from scripts.train_simsan import *
samples=discover_tess(Path("datasets/TESS"))+discover_crema(Path("datasets/CREMA-D"))+discover_emodb(Path("datasets/EmoDB"))
split=json.load(open("models/speech/simsan_split_manifest.json",encoding="utf8"))
sp=np.array([s.speaker for s in samples]); y=np.array([LABEL_TO_ID[s.label] for s in samples])
idx=np.flatnonzero(np.isin(sp,split["validation_speakers"]))
ds=SpectrogramDataset(Path("models/speech/simsan_logmel.npy"),idx,y,np.full(len(y),-1),False)
loader=DataLoader(ds,batch_size=96)
ck=torch.load("models/speech/simsan_best.pt",map_location="cpu",weights_only=True)
m=SIMSAN(7,ck["speaker_classes"]);m.load_state_dict(ck["state_dict"]);m.cuda()
metrics,yt,yp=loader_metrics(m,loader,torch.device("cuda"))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(yt,yp,labels=range(7)))
for source in ["CREMA-D","EmoDB","TESS"]:
 mask=np.array([samples[i].dataset==source for i in idx])
 if mask.any():
  print(source,len(yt[mask]),accuracy_score(yt[mask],yp[mask]),f1_score(yt[mask],yp[mask],average="macro"))
