import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
cache=np.load("models/speech/multidataset_features.npz",allow_pickle=True)
X,y=cache["X"],cache["y"].astype(str)
s=cache["speakers"].astype(str)
d=cache["datasets"].astype(str)
pair=(d=="TESS") & np.isin(y,["joy","surprise"])
train=pair & (s=="TESS:OAF")
test=pair & (s=="TESS:YAF")
variants={
 "full":X,
 "spread_only":np.c_[X[:,44:88],X[:,132:176]-X[:,88:132]],
 "drop_c0_duration":np.delete(X,[0,44,88,132,176],axis=1),
 "speaker_reduced":np.c_[X[:,1:44],X[:,45:88],X[:,133:176]-X[:,89:132]],
 "prosody_spread":np.c_[X[:,40:44],X[:,44:88],X[:,132:176]-X[:,88:132],X[:,176:177]],
}
for name,F in variants.items():
 for c in [0.1,1,3,10,30]:
    m=Pipeline([("s",StandardScaler()),("m",SVC(C=c,class_weight="balanced",probability=True))])
    m.fit(F[train],y[train]); p=m.predict(F[test])
    r=recall_score(y[test],p,labels=["joy","surprise"],average=None)
    print(name,c,round(accuracy_score(y[test],p),3),np.round(r,3),confusion_matrix(y[test],p,labels=["joy","surprise"]).tolist())
