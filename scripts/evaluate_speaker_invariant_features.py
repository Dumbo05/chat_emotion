import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cache=np.load("models/speech/multidataset_features.npz",allow_pickle=True)
X,y=cache["X"],cache["y"].astype(str)
speakers=cache["speakers"].astype(str)
datasets=cache["datasets"].astype(str)
common=np.unique(speakers[datasets!="TESS"])
train,temp=train_test_split(common,test_size=.30,random_state=42)
val,test=train_test_split(temp,test_size=.50,random_state=42)
train_mask=np.isin(speakers,np.r_[train,val])|(speakers=="TESS:OAF")
test_mask=np.isin(speakers,test)|(speakers=="TESS:YAF")
labels=["anger","disgust","fear","joy","sadness","surprise","neutral"]

variants={
 "full":X,
 "spread_only":np.c_[X[:,44:88],X[:,132:176]-X[:,88:132]],
 "drop_c0_duration":np.delete(X,[0,44,88,132,176],axis=1),
 "speaker_reduced":np.c_[X[:,1:44],X[:,45:88],X[:,133:176]-X[:,89:132]],
 "prosody_spread":np.c_[X[:,40:44],X[:,44:88],X[:,132:176]-X[:,88:132],X[:,176:177]],
}
for name,features in variants.items():
    model=Pipeline([("s",StandardScaler()),("m",SVC(C=3,class_weight="balanced"))])
    model.fit(features[train_mask],y[train_mask])
    pred=model.predict(features[test_mask])
    recalls=recall_score(y[test_mask],pred,labels=labels,average=None,zero_division=0)
    print(name,"acc",round(accuracy_score(y[test_mask],pred),4),
          "f1",round(f1_score(y[test_mask],pred,labels=labels,average="macro",zero_division=0),4),
          "recall",dict(zip(labels,np.round(recalls,3))))
