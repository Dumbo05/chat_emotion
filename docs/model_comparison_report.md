# 文本情绪识别模型对比报告

生成日期：2026-06-23
项目路径：`C:\Users\Lenovo\Desktop\del`

## 1. 评估说明

本报告汇总当前项目中已收到的文本情绪识别模型与 XLM-R 对比实验结果。所有模型统一输出七类情绪：

| 序号 | 英文标签 | 中文显示 |
|---:|---|---|
| 0 | anger | 愤怒 |
| 1 | disgust | 厌恶 |
| 2 | fear | 恐惧 |
| 3 | joy | 喜悦 |
| 4 | sadness | 悲伤 |
| 5 | surprise | 惊讶 |
| 6 | neutral | 中性 |

主要比较指标：

- `Accuracy`：整体准确率。
- `Macro-F1`：各类别 F1 的平均值，更适合类别不均衡的数据集。
- 项目选型优先看 `Test Macro-F1`，其次参考中文、英文分项表现。

## 2. 当前模型实验总览

| 模型/来源 | 本地路径 | 模型底座 | 实验类型 | Test Accuracy | Test Macro-F1 | English Macro-F1 | Chinese Macro-F1 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---|
| 组员 A 原始 XLM-R | `C:\Users\Lenovo\Desktop\del\A1\xlm-roberta` | XLM-RoBERTa Base | 单模型训练结果 | 0.6708 | 0.6065 | 0.6316 | 0.5173 | 中文分项表现优于 mBERT |
| 组员 B mBERT | `C:\Users\Lenovo\Desktop\del\member_b` | mBERT / bert-base-multilingual-cased | 单模型训练结果 | 0.6679 | 0.5960 | 0.6384 | 0.4907 | 英文分项略好，但整体和中文低于 A 组 XLM-R |
| XLM-R 对比实验最佳项 | `C:\Users\Lenovo\Desktop\del\xlm-roberta-experiments\BASELINE\BASELINE_baseline_seed123` | XLM-RoBERTa Base | 参数与 seed 对比实验 | 0.6812 | 0.6135 | 0.6544 | 0.5073 | 当前 XLM-R 对比实验中该配置整体指标最高 |

结论：
如果只看实验指标，`XLM-R + lr=2e-5 + max_length=128 + seed=123` 是目前最好的实验结果；组员 A 的原始 XLM-R 在中文分项上略占优势；组员 B 的 mBERT 在英文分项上略好，但整体 Macro-F1 低于 XLM-R。

## 3. 可用模型详细参数

| 模型 | Model Name | Best Validation Macro-F1 | Test Accuracy | Test Macro-F1 | English Accuracy | English Macro-F1 | Chinese Accuracy | Chinese Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 组员 A XLM-R | `xlm-roberta-base` | 0.6135 | 0.6708 | 0.6065 | 0.6896 | 0.6316 | 0.6447 | 0.5173 |
| 组员 B mBERT | `bert-base-multilingual-cased` | 0.6024 | 0.6679 | 0.5960 | 0.7009 | 0.6384 | 0.6220 | 0.4907 |

## 4. XLM-R 对比实验结果

实验目录：`C:\Users\Lenovo\Desktop\del\xlm-roberta-experiments`

| 排名 | 具体模型配置 | Learning Rate | Seed | Best Epoch | Val Macro-F1 | Test Accuracy | Test Macro-F1 | English Macro-F1 | Chinese Macro-F1 | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `XLM-R + lr=2e-5 + max_length=128 + seed=123` | 2e-5 | 123 | 4 | 0.6108 | 0.6812 | 0.6135 | 0.6544 | 0.5073 | 01:24:32 |
| 2 | `XLM-R + lr=1e-5 + max_length=128 + seed=456` | 1e-5 | 456 | 4 | 0.6094 | 0.6824 | 0.6133 | 0.6470 | 0.5104 | 01:30:57 |
| 3 | `XLM-R + lr=2e-5 + max_length=128 + seed=456` | 2e-5 | 456 | 3 | 0.6135 | 0.6847 | 0.6120 | 0.6421 | 0.5098 | 01:27:14 |
| 4 | `XLM-R + lr=1e-5 + max_length=128 + seed=123` | 1e-5 | 123 | 4 | 0.6123 | 0.6815 | 0.6118 | 0.6478 | 0.5087 | 01:32:09 |
| 5 | `XLM-R + lr=5e-6 + max_length=128 + seed=123` | 5e-6 | 123 | 4 | 0.6086 | 0.6793 | 0.6073 | 0.6378 | 0.5098 | 01:27:22 |
| 6 | `XLM-R + lr=5e-6 + max_length=128 + seed=456` | 5e-6 | 456 | 3 | 0.6058 | 0.6759 | 0.6051 | 0.6366 | 0.4999 | 01:23:54 |
| 7 | `XLM-R + lr=1e-5 + max_length=128 + seed=42` | 1e-5 | 42 | 4 | 0.6118 | 0.6781 | 0.6050 | 0.6292 | 0.5055 | 01:21:42 |
| 8 | `XLM-R + lr=2e-5 + max_length=128 + seed=42` | 2e-5 | 42 | 2 | 0.6109 | 0.6684 | 0.6050 | 0.6285 | 0.5208 | 01:24:40 |
| 9 | `XLM-R + lr=5e-6 + max_length=128 + seed=42` | 5e-6 | 42 | 4 | 0.6054 | 0.6781 | 0.6028 | 0.6307 | 0.4993 | 01:21:29 |

另外，`XLM-R + lr=5e-5 + max_length=128 + seed=42` 目录目前只有配置与 tokenizer 文件，没有 `metrics.json`，因此没有纳入排名。

## 5. 最佳配置判断

按 `Test Macro-F1` 排序：

| 名次 | 模型配置 | Test Accuracy | Test Macro-F1 | English Macro-F1 | Chinese Macro-F1 |
|---:|---|---:|---:|---:|---:|
| 1 | `XLM-R + lr=2e-5 + max_length=128 + seed=123` | 0.6812 | 0.6135 | 0.6544 | 0.5073 |
| 2 | 组员 A 原始 XLM-R | 0.6708 | 0.6065 | 0.6316 | 0.5173 |
| 3 | 组员 B mBERT | 0.6679 | 0.5960 | 0.6384 | 0.4907 |

数据分析结论：

1. `XLM-R + lr=2e-5 + max_length=128 + seed=123` 的整体测试表现最好，Test Macro-F1 达到 0.6135。
2. 组员 A 原始 XLM-R 的中文 Macro-F1 为 0.5173，高于 XLM-R 对比实验最佳项的 0.5073。
3. 组员 B mBERT 的英文 Macro-F1 为 0.6384，高于组员 A 原始 XLM-R，但低于 XLM-R 对比实验最佳项。
4. 从总体 Macro-F1 看，XLM-R 系列整体优于 mBERT。

## 6. 总体分析

从当前结果看，XLM-R 对比实验的上限略高于 mBERT。最佳 XLM-R 配置 `XLM-R + lr=2e-5 + max_length=128 + seed=123` 相比组员 B mBERT 提升如下：

| 对比项 | Test Accuracy | Test Macro-F1 | English Macro-F1 | Chinese Macro-F1 |
|---|---:|---:|---:|---:|
| `XLM-R + lr=2e-5 + max_length=128 + seed=123` | 0.6812 | 0.6135 | 0.6544 | 0.5073 |
| 组员 B mBERT | 0.6679 | 0.5960 | 0.6384 | 0.4907 |
| 差值 | +0.0133 | +0.0175 | +0.0160 | +0.0166 |

最佳 XLM-R 配置相比组员 A 原始 XLM-R：

| 对比项 | Test Accuracy | Test Macro-F1 | English Macro-F1 | Chinese Macro-F1 |
|---|---:|---:|---:|---:|
| `XLM-R + lr=2e-5 + max_length=128 + seed=123` | 0.6812 | 0.6135 | 0.6544 | 0.5073 |
| 组员 A 原始 XLM-R | 0.6708 | 0.6065 | 0.6316 | 0.5173 |
| 差值 | +0.0104 | +0.0070 | +0.0228 | -0.0100 |

说明：

1. `XLM-R + lr=2e-5 + max_length=128 + seed=123` 的综合能力更强，尤其英文分项提升明显。
2. 组员 A 原始 XLM-R 在中文分项上更好，说明中文数据表现存在 seed 或训练过程波动。
3. 如果论文或答辩需要选择“最佳总体模型”，建议选择 `XLM-R + lr=2e-5 + max_length=128 + seed=123`。
4. 如果强调中文情绪识别效果，可以补充说明组员 A 原始 XLM-R 的中文 Macro-F1 更高。
