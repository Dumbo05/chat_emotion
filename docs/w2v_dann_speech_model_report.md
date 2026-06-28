# Wav2Vec2-DANN 语音情绪识别模型报告补充

生成日期：2026-06-28
部署程序：`dist/EmotionRecognition.exe`
服务器复现目录：`/data/reproduce_w2v`
本地模型目录：`models/speech/w2v_dann`

## 1. 模型概述

本次语音模块采用 Wav2Vec2 预训练声学表征 + DANN 域对抗训练的方案，目标是在 TESS 数据集的跨说话人场景下提升泛化能力。模型以 `facebook/wav2vec2-base` 作为语音编码器，在其上加入情绪分类分支和说话人域分类分支，通过 Gradient Reversal Layer（GRL）削弱模型对说话人身份的依赖，使情绪特征更加稳定。

与普通声学特征或单一分类器相比，该方法的优势在于可以利用 Wav2Vec2 已学习到的大规模语音表征，同时用域对抗约束减少训练说话人与测试说话人之间的差异。该设计适合 TESS 这类干净录音、类别明确、说话人数量有限的数据集。

## 2. 数据与任务设置

数据集使用 TESS Toronto emotional speech set。实验按说话人独立协议进行划分：OAF 作为源域训练数据，YAF 作为目标域验证和测试数据。情绪类别共 7 类，包括 angry、disgust、fear、happy、neutral、pleasant_surprise、sad。部署到系统后，`pleasant_surprise` 统一映射为界面中的“惊讶”类别。

输入音频统一重采样到 16 kHz 单声道，长度固定为 3 秒，即 48,000 个采样点。短音频进行零填充，长音频进行截断，并对波形进行标准化处理。训练阶段对源域音频使用轻量增强，以提升模型对音量、扰动和录音差异的鲁棒性。

## 3. 网络结构

模型主干为 `facebook/wav2vec2-base`。为了避免小数据集上过拟合，冻结 Wav2Vec2 的前端特征提取器，并冻结前 8 层 Transformer，仅微调较高层语义特征。

模型使用 Wav2Vec2 多层隐藏状态加权融合，不只依赖最后一层输出。融合后的帧级特征送入 Self-Attentive Statistical Pooling 模块，得到加权均值和加权标准差拼接后的句级表示。该表示再经过情绪分类头输出 7 类情绪概率。

域对抗分支以 Wav2Vec2 最后一层隐藏状态的平均池化结果作为输入，先经过 GRL，再进入说话人域分类头，预测样本来自源域还是目标域。GRL 在反向传播时反转梯度，使主干学习到更不依赖说话人的特征。

主要结构如下：

| 模块 | 配置 |
| --- | --- |
| 预训练主干 | `facebook/wav2vec2-base` |
| 输入长度 | 3 秒，16 kHz，48,000 samples |
| 情绪类别数 | 7 |
| 冻结策略 | 冻结 feature extractor 和前 8 层 Transformer |
| 特征融合 | 13 层 hidden states 可学习加权融合 |
| 池化方式 | Self-Attentive Statistical Pooling，均值 + 标准差 |
| 情绪分类头 | Linear 1536 -> 256 -> 7，GELU + Dropout |
| 域分类头 | GRL + Linear 768 -> 256 -> 2 |

## 4. 训练参数

训练采用 AdamW 优化器，并对主干和分类头使用不同学习率。主干使用较小学习率保持预训练表征稳定，新增分类层使用较大学习率加快收敛。

| 参数 | 数值 |
| --- | --- |
| batch size | 8 |
| 最大 epoch | 40 |
| early stopping patience | 20 |
| optimizer | AdamW |
| backbone learning rate | 1e-5 |
| head learning rate | 1e-3 |
| weight decay | 1e-4 |
| label smoothing | 0.1 |
| domain loss weight | 0.08 |
| scheduler | Cosine Annealing |
| eta_min | 1e-6 |
| GRL alpha | `2 / (1 + exp(-10p)) - 1` |
| mixed precision | AMP |
| gradient clipping | 1.0 |

## 5. 服务器复现结果

服务器硬件为 NVIDIA GeForce RTX 3090 24GB，Python 3.12 环境下复现评估成功。测试集包含 1120 条 YAF 样本，复现得到：

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 89.38% |
| Weighted Recall | 89.38% |
| Weighted F1-score | 88.29% |

原项目报告中的指标约为 Accuracy 89.55%、Weighted F1-score 88.47%。本次复现结果与原结果只相差约 0.17 个百分点，属于正常复现误差范围，说明模型权重、数据划分和推理流程基本一致。

从分类报告看，disgust、neutral、sad 等类别表现较稳定；happy 类召回率较低，主要被 angry 或 fear 等类别混淆。这说明模型在 TESS 的干净跨说话人设置下整体有效，但对相近情绪或表达强度变化仍有不足。

## 6. 合理性分析

89%左右的准确率在该实验条件下是合理的。TESS 数据集录音质量高、背景噪声少、情绪表达较夸张，且类别标签清晰，因此模型可以取得较高分数。同时，测试协议是 OAF 到 YAF 的跨说话人评估，能够验证一定的说话人泛化能力。

但该结果不应直接等同于真实开放场景准确率。真实用户语音会包含普通话或方言口音、环境噪声、麦克风差异、自然弱情绪、语速变化和非标准句子长度，这些因素都会显著降低模型效果。因此报告中建议表述为“在 TESS 干净语音跨说话人测试集上达到 89.38% 准确率”，而不是泛化为“真实场景准确率 89.38%”。

## 7. 系统部署方式

为了降低桌面程序体积和运行依赖，最终没有在 exe 中直接打包 PyTorch 和 Transformers，而是将训练好的 Wav2Vec2-DANN 模型导出为 ONNX：

| 文件 | 说明 |
| --- | --- |
| `models/speech/w2v_dann/w2v_dann_opt.onnx` | ONNX 主模型结构 |
| `models/speech/w2v_dann/w2v_dann_opt.onnx.data` | ONNX 外部权重数据 |
| `models/speech/w2v_dann/server_eval_metrics.json` | 服务器复现指标记录 |

应用端使用 ONNX Runtime 推理。语音识别流程优先调用 Wav2Vec2-DANN 模型，如果该模型不可用，再回退到旧的 WavLM-SIMSAN 或传统模型。最终程序已经重新打包到：

`C:\Users\Lenovo\Desktop\del\dist\EmotionRecognition.exe`

本地验证结果显示，文本识别 smoke test 和语音识别 smoke test 均可正常运行。语音测试样本 `YAF_back_angry.wav` 被识别为“愤怒”，模型名称显示为 `Wav2Vec2-DANN Optimized`。

## 8. 报告建议表述

可在正式实验报告中使用以下简短描述：

> 语音情绪识别模块采用 Wav2Vec2-DANN 方法。模型以 `facebook/wav2vec2-base` 为预训练声学编码器，冻结前端特征提取器和前 8 层 Transformer，并通过多层隐藏状态加权融合和 Self-Attentive Statistical Pooling 获取句级语音表示。在情绪分类分支之外，引入 Gradient Reversal Layer 构建说话人域分类分支，通过域对抗训练降低模型对说话人身份的依赖。模型在 TESS 数据集 OAF 到 YAF 的跨说话人测试协议下达到 89.38% Accuracy、89.38% Weighted Recall 和 88.29% Weighted F1-score，结果与原项目报告指标接近，说明复现成功。需要注意的是，该指标反映的是干净录音数据集上的性能，真实开放环境中可能受噪声、口音、麦克风和自然情绪强度影响而下降。
