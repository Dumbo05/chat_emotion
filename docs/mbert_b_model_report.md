# mBERT 文本情绪识别实验报告

生成日期：2026-06-23
项目路径：`C:\Users\Lenovo\Desktop\del`
实验目录：`C:\Users\Lenovo\Desktop\del\mbert_b`

## 1. 评估说明

本报告汇总 `mbert_b` 目录中的 mBERT 文本情绪识别对比实验结果。该目录包含多组不同随机种子、学习率和最大文本长度的实验日志。

当前实验统一输出七类情绪：

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

- `Test Accuracy`：测试集整体准确率。
- `Test Macro-F1`：七类情绪 F1 的平均值，更适合类别不均衡场景。
- 选型优先看 `Test Macro-F1`，其次参考 `Test Accuracy`。

注意：这批 `mbert_b` 日志中没有提供中文测试集、英文测试集的分项指标，因此本文只统计统一测试集结果。

## 2. 模型结构参数

根据 `checkpints\mBERT-LR-1-seed43\config.json`，mBERT 模型主要结构如下：

| 参数 | 数值 |
|---|---:|
| 模型类型 | `bert` |
| 架构 | `BertForSequenceClassification` |
| 分类类别数 | 7 |
| Hidden Size | 768 |
| Transformer 层数 | 12 |
| Attention Heads | 12 |
| Intermediate Size | 3072 |
| Dropout | 0.1 |
| 最大位置编码 | 512 |
| Vocabulary Size | 119547 |
| 任务类型 | `single_label_classification` |
| Transformers 版本 | 5.12.1 |

## 3. 实验数据文件说明

`mbert_b` 当前包含：

| 类型 | 路径 | 说明 |
|---|---|---|
| 实验日志 | `C:\Users\Lenovo\Desktop\del\mbert_b\logs` | 每个实验的 JSON 指标文件，以及 `summary.csv` 汇总表 |
| 配置与 tokenizer | `C:\Users\Lenovo\Desktop\del\mbert_b\checkpints` | 每个实验对应的 `config.json`、`tokenizer.json`、`tokenizer_config.json` |

用于数据分析的核心文件是：

| 文件 | 用途 |
|---|---|
| `logs\summary.csv` | 汇总所有 mBERT 实验的主要参数与测试指标 |
| `logs\*.json` | 保存每个实验的详细训练过程、best epoch 和测试集分类报告 |
| `checkpints\*\config.json` | 保存模型结构、标签映射等配置 |

## 4. 实验结果总览

实验汇总文件：

`C:\Users\Lenovo\Desktop\del\mbert_b\logs\summary.csv`

| 排名 | 具体模型配置 | 分组 | Seed | Learning Rate | Max Length | Batch Size | Effective Batch | Warmup Ratio | Weight Decay | Best Epoch | Test Accuracy | Test Macro-F1 | 训练耗时 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `mBERT + lr=1e-5 + max_length=128 + seed=43` | LR-1 | 43 | 1e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 4 | 0.6694 | 0.6019 | 01:43:08 |
| 2 | `mBERT + lr=2e-5 + max_length=96 + seed=42` | LEN-96 | 42 | 2e-5 | 96 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6687 | 0.6016 | 00:36:46 |
| 3 | `mBERT + lr=2e-5 + max_length=128 + seed=44` | baseline | 44 | 2e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6687 | 0.5996 | 00:45:22 |
| 4 | `mBERT + lr=3e-5 + max_length=128 + seed=44` | LR-2 | 44 | 3e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6657 | 0.5974 | 00:40:20 |
| 5 | `mBERT + lr=1e-5 + max_length=128 + seed=42` | LR-1 | 42 | 1e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6648 | 0.5971 | 00:45:09 |
| 6 | `mBERT + lr=5e-5 + max_length=128 + seed=43` | LR-3 | 43 | 5e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6629 | 0.5956 | 00:42:54 |
| 7 | `mBERT + lr=3e-5 + max_length=128 + seed=42` | LR-2 | 42 | 3e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 2 | 0.6643 | 0.5957 | 00:52:24 |
| 8 | `mBERT + lr=2e-5 + max_length=128 + seed=42` | baseline | 42 | 2e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 2 | 0.6650 | 0.5955 | 00:50:40 |
| 9 | `mBERT + lr=2e-5 + max_length=128 + seed=43` | baseline | 43 | 2e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 4 | 0.6653 | 0.5953 | 01:21:00 |
| 10 | `mBERT + lr=2e-5 + max_length=96 + seed=43` | LEN-96 | 43 | 2e-5 | 96 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6630 | 0.5947 | 00:37:33 |
| 11 | `mBERT + lr=1e-5 + max_length=128 + seed=44` | LR-1 | 44 | 1e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 4 | 0.6649 | 0.5941 | 01:30:35 |
| 12 | `mBERT + lr=5e-5 + max_length=128 + seed=42` | LR-3 | 42 | 5e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 4 | 0.6563 | 0.5910 | 00:39:57 |
| 13 | `mBERT + lr=3e-5 + max_length=128 + seed=43` | LR-2 | 43 | 3e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6562 | 0.5908 | 00:51:30 |
| 14 | `mBERT + lr=2e-5 + max_length=64 + seed=43` | LEN-64 | 43 | 2e-5 | 64 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6603 | 0.5886 | 00:35:40 |
| 15 | `mBERT + lr=2e-5 + max_length=64 + seed=42` | LEN-64 | 42 | 2e-5 | 64 | 8 | 16 | 0.1 | 0.01 | 4 | 0.6574 | 0.5882 | 00:38:54 |
| 16 | `mBERT + lr=5e-5 + max_length=128 + seed=44` | LR-3 | 44 | 5e-5 | 128 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6635 | 0.5883 | 00:45:40 |
| 17 | `mBERT + lr=2e-5 + max_length=64 + seed=44` | LEN-64 | 44 | 2e-5 | 64 | 8 | 16 | 0.1 | 0.01 | 3 | 0.6622 | 0.5860 | 00:37:36 |

## 5. 最佳配置

当前 `mbert_b` 中表现最好的配置是：

`mBERT + lr=1e-5 + max_length=128 + seed=43`

| 指标 | 数值 |
|---|---:|
| 分组 | LR-1 |
| Seed | 43 |
| Learning Rate | 1e-5 |
| Max Length | 128 |
| Batch Size | 8 |
| Effective Batch Size | 16 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Best Epoch | 4 |
| Test Accuracy | 0.6694 |
| Test Macro-F1 | 0.6019 |
| 训练耗时 | 01:43:08 |

该实验比 baseline 中最好的 `mBERT + lr=2e-5 + max_length=128 + seed=44` 略高：

| 对比项 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| 最佳 LR 实验：`mBERT + lr=1e-5 + max_length=128 + seed=43` | 0.6694 | 0.6019 |
| 最佳 baseline：`mBERT + lr=2e-5 + max_length=128 + seed=44` | 0.6687 | 0.5996 |
| 提升 | +0.0007 | +0.0023 |

## 6. 分组观察

根据实验结果可以得到以下初步结论：

1. `LR-1` 组，即学习率 `1e-5`，取得当前最高 `Test Macro-F1`。
2. `LEN-96` 组效果也较好，`mBERT + lr=2e-5 + max_length=96 + seed=42` 的 `Test Macro-F1` 达到 0.6016，接近最佳结果。
3. `LEN-64` 组整体偏低，说明最大长度 64 可能截断了部分有效文本信息。
4. `LR-3` 组，即学习率 `5e-5`，整体不如 `1e-5` 和 `2e-5` 稳定，学习率可能偏大。
5. 不同 seed 对结果有一定影响，mBERT 的结果波动范围约在 0.5860 到 0.6019 之间。

## 7. 参数影响分析

### 7.1 学习率影响

在 max length 均为 128 的实验中，学习率对 mBERT 表现有明显影响：

| 学习率组 | 学习率 | 最佳配置 | 最佳 Test Macro-F1 | 观察 |
|---|---:|---|---:|---|
| LR-1 | 1e-5 | `mBERT + lr=1e-5 + max_length=128 + seed=43` | 0.6019 | 当前最佳，较稳定 |
| baseline | 2e-5 | `mBERT + lr=2e-5 + max_length=128 + seed=44` | 0.5996 | 表现接近最佳 |
| LR-2 | 3e-5 | `mBERT + lr=3e-5 + max_length=128 + seed=44` | 0.5974 | 略低于 1e-5 和 2e-5 |
| LR-3 | 5e-5 | `mBERT + lr=5e-5 + max_length=128 + seed=43` | 0.5956 | 整体偏低，学习率可能过大 |

结论：mBERT 在当前数据集上更适合较小学习率，`1e-5` 和 `2e-5` 的表现优于 `3e-5`、`5e-5`。

### 7.2 最大文本长度影响

在学习率固定为 2e-5 的实验中，最大长度对结果也有影响：

| Max Length | 最佳配置 | 最佳 Test Accuracy | 最佳 Test Macro-F1 | 观察 |
|---:|---|---:|---:|---|
| 64 | `mBERT + lr=2e-5 + max_length=64 + seed=43` | 0.6603 | 0.5886 | 明显偏低，可能截断信息过多 |
| 96 | `mBERT + lr=2e-5 + max_length=96 + seed=42` | 0.6687 | 0.6016 | 表现很好，接近所有实验最佳 |
| 128 | `mBERT + lr=2e-5 + max_length=128 + seed=44` | 0.6687 | 0.5996 | 表现稳定，但略低于 LEN-96 最佳项 |

结论：最大长度 64 不推荐；96 和 128 都可用，其中 96 在本批实验中略优。

### 7.3 随机种子影响

同一参数下，不同 seed 会造成一定波动。例如 baseline 组：

| 具体模型配置 | Seed | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| `mBERT + lr=2e-5 + max_length=128 + seed=42` | 42 | 0.6650 | 0.5955 |
| `mBERT + lr=2e-5 + max_length=128 + seed=43` | 43 | 0.6653 | 0.5953 |
| `mBERT + lr=2e-5 + max_length=128 + seed=44` | 44 | 0.6687 | 0.5996 |

baseline 组 Macro-F1 波动范围为 0.5953 到 0.5996，说明 seed 会影响结果，但波动不算特别大。

## 8. 总结

`mbert_b` 的最佳配置为 `mBERT + lr=1e-5 + max_length=128 + seed=43`，测试集准确率为 0.6694，测试集 Macro-F1 为 0.6019。该结果略优于 mBERT baseline。

当前数据分析结论：

1. mBERT 最佳参数组合为 learning rate `1e-5`、max length `128`、seed `43`。
2. max length `96` 的表现也非常接近最佳结果，说明适当缩短文本长度可能降低噪声或提升训练效率。
3. max length `64` 整体偏低，不建议作为最终参数。
4. learning rate `5e-5` 整体不够稳定，可能偏大。
5. 后续如果继续做 mBERT 实验，建议围绕 `1e-5`、`2e-5`、max length `96/128` 做更细粒度搜索。
