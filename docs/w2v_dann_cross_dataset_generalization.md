# Wav2Vec2-DANN 语音模型跨数据集泛化测试

生成日期：2026-06-28
测试模型：`models/speech/w2v_dann/w2v_dann_opt.onnx`
测试脚本：`scripts/speech/evaluate_w2v_dann_cross_dataset.py`

## 1. 测试目的

此前模型在 TESS 跨说话人测试集上达到 89.38% Accuracy、88.29% Weighted F1-score。该结果说明模型能够较好拟合 TESS 的干净录音和跨说话人划分，但不能直接代表真实开放场景表现。因此，本次使用 CREMA-D 和 EmoDB 两个外部语音情绪数据集测试模型的跨数据集泛化能力。

## 2. 测试设置

测试时不重新训练模型，只使用当前打包进程序的 Wav2Vec2-DANN ONNX 模型进行推理。不同数据集的标签被映射到系统统一情绪空间：

| 数据集 | 使用类别 |
| --- | --- |
| CREMA-D | anger、disgust、fear、joy、neutral、sadness |
| EmoDB | anger、disgust、fear、joy、neutral、sadness |

CREMA-D 本轮采用每类 50 条样本的均衡抽样，共 300 条。EmoDB 跳过 boredom 类，因为当前系统没有对应类别，其余可映射样本共 454 条。

## 3. 结果汇总

| 数据集 | 样本数 | Accuracy | Weighted Recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| TESS（同源复现测试） | 1120 | 89.38% | 89.38% | 88.29% |
| CREMA-D（每类 50 条） | 300 | 21.33% | 21.33% | 17.71% |
| EmoDB（可映射全量） | 454 | 39.43% | 39.43% | 40.62% |

## 4. 每类表现

### CREMA-D

| 类别 | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| anger | 50 | 24.4% | 22.0% | 23.2% |
| disgust | 50 | 26.7% | 72.0% | 38.9% |
| fear | 50 | 50.0% | 2.0% | 3.8% |
| joy | 50 | 7.1% | 2.0% | 3.1% |
| sadness | 50 | 23.1% | 12.0% | 15.8% |
| neutral | 50 | 26.5% | 18.0% | 21.4% |

CREMA-D 上模型明显偏向预测 disgust，同时 fear 和 joy 几乎无法正确召回，说明模型在英语多人、多语句、多强度数据集上的泛化能力较弱。

### EmoDB

| 类别 | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| anger | 127 | 45.3% | 37.8% | 41.2% |
| disgust | 46 | 32.7% | 37.0% | 34.7% |
| fear | 69 | 46.6% | 39.1% | 42.5% |
| joy | 71 | 39.7% | 35.2% | 37.3% |
| sadness | 62 | 47.6% | 32.3% | 38.5% |
| neutral | 79 | 40.8% | 53.2% | 46.2% |

EmoDB 上模型表现优于 CREMA-D，但仍远低于 TESS。原因可能包括语言差异、录音风格差异、说话人分布差异、情绪表达方式差异，以及 TESS 模型训练时只见过 OAF/YAF 这类高度表演化、干净环境语音。

## 5. 结论

该模型在 TESS 数据集内复现结果合理，但跨数据集泛化能力不足。它更适合作为“受控语音情绪识别实验模型”，不宜直接宣称具备真实开放场景的高准确率。

建议报告中使用如下表述：

> Wav2Vec2-DANN 语音模型在 TESS 跨说话人测试集上取得 89.38% Accuracy，说明其在同源干净语音条件下具有较好识别能力。但在 CREMA-D 和 EmoDB 外部数据集上的准确率分别下降至 21.33% 和 39.43%，表明模型仍存在明显数据集依赖，真实场景泛化能力有限。后续可通过多数据集联合训练、跨语料域对抗、自监督特征微调和真实环境噪声增强进一步提升泛化能力。

## 6. 复现实验命令

```powershell
.\.venv\Scripts\python.exe .\scripts\speech\evaluate_w2v_dann_cross_dataset.py --dataset-path .\datasets\CREMA-D --dataset crema-d --max-per-class 50
```

```powershell
.\.venv\Scripts\python.exe .\scripts\speech\evaluate_w2v_dann_cross_dataset.py --dataset-path .\datasets\EmoDB --dataset emodb
```

输出目录：

`server-results/w2v_dann_cross_dataset`
