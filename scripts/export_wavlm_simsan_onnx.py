import os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
sys.path.insert(0,'.')
from emotion_app.audio_features import _read_audio

os.environ['HF_HOME']=str(Path('.hf_cache').resolve())
out=Path('models/speech/wavlm_simsan_encoder.onnx')
base=AutoModel.from_pretrained('microsoft/wavlm-base-plus').eval()
class LayerStats(torch.nn.Module):
 def __init__(self,m): super().__init__(); self.m=m
 def forward(self,input_values):
  h=self.m(input_values,output_hidden_states=True,return_dict=True).hidden_states[10]
  return torch.cat((h.mean(1),h.std(1,unbiased=False)),dim=1)
model=LayerStats(base).eval(); dummy=torch.randn(1,32000)
with torch.inference_mode(): reference=model(dummy).numpy()
torch.onnx.export(model,(dummy,),out,input_names=['input_values'],output_names=['layer10_stats'],dynamic_axes={'input_values':{1:'audio_samples'},'layer10_stats':{0:'batch'}},opset_version=17,dynamo=False)
import onnxruntime as ort
session=ort.InferenceSession(str(out),providers=['CPUExecutionProvider']); actual=session.run(None,{'input_values':dummy.numpy()})[0]
print({'path':str(out),'bytes':out.stat().st_size,'max_abs_diff':float(np.max(np.abs(reference-actual))),'mean_abs_diff':float(np.mean(np.abs(reference-actual)))})
