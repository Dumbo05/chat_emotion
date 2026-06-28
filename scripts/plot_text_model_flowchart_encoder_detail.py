from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','Arial','DejaVu Sans']
plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42; plt.rcParams['font.size']=8
OUT=Path(__file__).resolve().parents[1]/'docs'/'research-figures'/'text_model_flowchart_encoder_detail_cn'
C={'ink':'#25313C','muted':'#66727D','line':'#708090','input':'#EAF1F8','input_edge':'#6D8FB3','embed':'#EAF5E6','embed_edge':'#6F9B63','enc_bg':'#EAF2FB','enc_edge':'#2867A6','attn':'#D8E9F8','attn_edge':'#3E78B2','ffn':'#E4E0F4','ffn_edge':'#7668A8','norm':'#F8E9D6','norm_edge':'#B67A36','head':'#FBECDD','head_edge':'#C57D32','out':'#F2EAF5','out_edge':'#9A5F9C','white':'#FFFFFF'}

def rbox(ax,x,y,w,h,face,edge,lw=1.1,rad=.012,z=2):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.006,rounding_size={rad}',facecolor=face,edgecolor=edge,linewidth=lw,zorder=z); ax.add_patch(p); return p

def arrow(ax,a,b,color=None,lw=1.15,style='-|>',z=6):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,linewidth=lw,color=color or C['line'],shrinkA=2,shrinkB=2,zorder=z))

def txt(ax,x,y,s,size=7,weight='normal',color=None,ha='center',va='center',z=8,linespacing=1.2):
    ax.text(x,y,s,fontsize=size,fontweight=weight,color=color or C['ink'],ha=ha,va=va,zorder=z,linespacing=linespacing)

fig=plt.figure(figsize=(210/25.4,122/25.4),facecolor='white')
ax=fig.add_axes([.025,.035,.95,.93]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,.5,.974,'文本模态情感识别模型架构',13,'bold')
txt(ax,.5,.935,'以 XLM-RoBERTa Encoder 为核心的中英文统一七分类流程',7.5,color=C['muted'])

rbox(ax,.015,.815,.97,.095,C['input'],C['input_edge'],1.15,.012)
txt(ax,.035,.875,'输入层',8.2,'bold',ha='left')
rbox(ax,.145,.84,.18,.043,C['white'],C['input_edge'],.8,.007); txt(ax,.235,.862,'中文 / 英文原始文本',6.8,'bold')
arrow(ax,(.33,.862),(.385,.862),C['input_edge'])
rbox(ax,.39,.84,.25,.043,C['white'],C['input_edge'],.8,.007); txt(ax,.515,.862,'SentencePiece 子词切分',6.8,'bold')
arrow(ax,(.645,.862),(.70,.862),C['input_edge'])
rbox(ax,.705,.84,.21,.043,C['white'],C['input_edge'],.8,.007); txt(ax,.81,.862,'input_ids + attention_mask',6.5,'bold')
txt(ax,.515,.823,'去除首尾空白  ·  添加 <s> / </s>  ·  截断至 128 tokens',6.1,color=C['muted'])
arrow(ax,(.515,.815),(.515,.785),C['input_edge'])

rbox(ax,.015,.285,.745,.49,C['enc_bg'],C['enc_edge'],1.8,.016)
txt(ax,.035,.742,'XLM-RoBERTa Encoder（核心上下文编码器）',9.5,'bold',C['enc_edge'],ha='left')
txt(ax,.735,.742,'12 layers  ·  12 heads  ·  hidden size 768',6.2,'bold',C['enc_edge'],ha='right')

rbox(ax,.035,.365,.16,.32,C['embed'],C['embed_edge'],1.1,.012)
txt(ax,.115,.654,'输入嵌入',7.6,'bold',C['embed_edge'])
rbox(ax,.055,.575,.12,.047,C['white'],C['embed_edge'],.75,.006); txt(ax,.115,.598,'Token Embedding',6.1,'bold')
rbox(ax,.055,.505,.12,.047,C['white'],C['embed_edge'],.75,.006); txt(ax,.115,.528,'Position Embedding',6.1,'bold')
txt(ax,.115,.562,'+',12,'bold',C['embed_edge'])
rbox(ax,.055,.425,.12,.048,C['white'],C['embed_edge'],.75,.006); txt(ax,.115,.449,'LayerNorm + Dropout',5.9,'bold')
txt(ax,.115,.395,'词表 250,002  ·  向量维度 768',5.6,color=C['muted'])
arrow(ax,(.195,.525),(.235,.525),C['embed_edge'])

stack_x=.225; stack_y=.395; stack_w=.13; stack_h=.245
for i in range(5):
    ax.add_patch(FancyBboxPatch((stack_x+i*.010,stack_y+i*.014),stack_w,stack_h,boxstyle='round,pad=.004,rounding_size=.010',facecolor='#C8DDF2',edgecolor=C['enc_edge'],linewidth=.75,alpha=.42+i*.10,zorder=3+i))
txt(ax,.31,.625,'Transformer Encoder',7.1,'bold',C['enc_edge'])
txt(ax,.31,.586,'Layer 1',6.1,'bold'); txt(ax,.31,.525,'...' ,15,'bold',C['enc_edge']); txt(ax,.31,.468,'Layer 12',6.1,'bold')
txt(ax,.31,.421,'逐层双向上下文建模',5.7,color=C['muted'])
arrow(ax,(.38,.525),(.415,.525),C['enc_edge'])

rbox(ax,.41,.335,.325,.36,C['white'],C['enc_edge'],1.15,.012)
txt(ax,.572,.67,'单个 Transformer Encoder Layer（放大）',7.4,'bold',C['enc_edge'])
rbox(ax,.435,.525,.275,.105,C['attn'],C['attn_edge'],1.0,.009)
txt(ax,.572,.606,'多头自注意力 Multi-Head Self-Attention',6.5,'bold',C['attn_edge'])
txt(ax,.572,.579,'Q = H WQ   ·   K = H WK   ·   V = H WV',5.5)
txt(ax,.572,.548,'Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V',5.45,'bold')
rbox(ax,.465,.476,.215,.032,C['norm'],C['norm_edge'],.75,.005); txt(ax,.572,.492,'残差连接 Residual + LayerNorm',5.7,'bold',C['norm_edge'])
arrow(ax,(.572,.525),(.572,.508),C['attn_edge'])
rbox(ax,.435,.375,.275,.078,C['ffn'],C['ffn_edge'],1.0,.009)
txt(ax,.572,.429,'前馈网络 Feed-Forward Network',6.4,'bold',C['ffn_edge'])
txt(ax,.572,.398,'Linear 768→3072  ·  GELU  ·  Linear 3072→768',5.6,'bold')
arrow(ax,(.572,.476),(.572,.453),C['norm_edge'])
rbox(ax,.465,.34,.215,.025,C['norm'],C['norm_edge'],.75,.005); txt(ax,.572,.353,'残差连接 + LayerNorm  ·  Dropout p=0.1',5.4,'bold',C['norm_edge'])
arrow(ax,(.572,.375),(.572,.365),C['ffn_edge'])
for x,label in [(.448,'Q'),(.478,'K'),(.508,'V')]:
    ax.add_patch(Circle((x,.648),.012,facecolor=C['attn'],edgecolor=C['attn_edge'],linewidth=.7,zorder=7)); txt(ax,x,.648,label,5.6,'bold',C['attn_edge'])
arrow(ax,(.43,.648),(.435,.606),C['attn_edge'],.8)

rbox(ax,.235,.305,.145,.050,C['white'],C['enc_edge'],.9,.007); txt(ax,.307,.33,'输出 <s> 上下文向量',6.2,'bold',C['enc_edge'])
arrow(ax,(.38,.33),(.41,.33),C['enc_edge']); txt(ax,.46,.309,'h<s> in R^768',6,'bold',C['enc_edge'])

rbox(ax,.78,.285,.205,.49,C['head'],C['head_edge'],1.35,.015)
txt(ax,.80,.742,'分类与输出',8.5,'bold',C['head_edge'],ha='left')
arrow(ax,(.76,.525),(.795,.525),C['head_edge'])
rbox(ax,.805,.635,.155,.055,C['white'],C['head_edge'],.85,.007); txt(ax,.882,.663,'<s> 特征向量（768-d）',6.2,'bold')
arrow(ax,(.882,.635),(.882,.595),C['head_edge'])
rbox(ax,.805,.54,.155,.055,C['white'],C['head_edge'],.85,.007); txt(ax,.882,.568,'Dropout + Linear',6.5,'bold')
txt(ax,.882,.526,'768 → 7 logits',5.8,color=C['muted'])
arrow(ax,(.882,.54),(.882,.485),C['head_edge'])
rbox(ax,.805,.435,.155,.05,C['white'],C['head_edge'],.85,.007); txt(ax,.882,.46,'Softmax 概率归一化',6.2,'bold')
arrow(ax,(.882,.435),(.882,.405),C['head_edge'])
rbox(ax,.805,.315,.155,.085,C['out'],C['out_edge'],.95,.008)
txt(ax,.882,.378,'七类情绪空间',6.4,'bold',C['out_edge'])
txt(ax,.882,.346,'anger · disgust · fear · joy',5.3); txt(ax,.882,.325,'sadness · surprise · neutral',5.3)

arrow(ax,(.882,.285),(.882,.25),C['out_edge'])
rbox(ax,.015,.12,.97,.115,'#F7F8FA','#9AA3AB',1.05,.012)
txt(ax,.035,.198,'输出层',8.2,'bold',ha='left')
rbox(ax,.20,.15,.22,.045,C['out'],C['out_edge'],.85,.006); txt(ax,.31,.173,'预测情绪 = argmax(probabilities)',6.5,'bold',C['out_edge'])
arrow(ax,(.425,.173),(.48,.173),C['line'])
rbox(ax,.485,.15,.19,.045,C['white'],'#9AA3AB',.8,.006); txt(ax,.58,.173,'置信度 + 七类概率分布',6.3,'bold')
arrow(ax,(.68,.173),(.735,.173),C['line'])
rbox(ax,.74,.15,.18,.045,C['white'],'#9AA3AB',.8,.006); txt(ax,.83,.173,'RecognitionResult',6.3,'bold')
txt(ax,.5,.095,'推理：本地权重  ·  eval 模式  ·  inference_mode（无梯度）  ·  CPU / GPU 自动选择',6.3,color=C['muted'])
ax.annotate('Encoder 占据主要计算与表征能力',xy=(.585,.71),xytext=(.34,.765),fontsize=5.8,color=C['enc_edge'],ha='center',va='center',arrowprops=dict(arrowstyle='->',color=C['enc_edge'],lw=.8),zorder=10)

OUT.parent.mkdir(parents=True,exist_ok=True)
for ext,dpi in [('svg',None),('pdf',None),('png',300),('tiff',600)]: fig.savefig(OUT.with_suffix('.'+ext),dpi=dpi,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
print(OUT)
