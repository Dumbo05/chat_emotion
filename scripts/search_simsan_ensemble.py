import numpy as np
from sklearn.metrics import accuracy_score, f1_score
d=np.load('models/speech/simsan_validation_logits.npz'); y=d['y']; z0=d['best']; z1=d['balanced']
best=(-1,None)
for t0 in np.linspace(.3,2.5,45):
 for t1 in np.linspace(.3,2.5,45):
  for a in np.linspace(0,1,101):
   p=(a*z0/t0+(1-a)*z1/t1).argmax(1)
   acc=accuracy_score(y,p)
   if acc>best[0]: best=(acc,(float(a),float(t0),float(t1),f1_score(y,p,average='macro')))
print(best)
