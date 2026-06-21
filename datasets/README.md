# 数据集目录

本项目把训练、评估所需数据统一放在 `datasets/` 下，避免数据散落在工程根目录。受数据许可和仓库体积限制，第三方原始数据默认不提交到 Git；目录位置、来源和用途如下。

| 目录 | 数据集 / 内容 | 用途 | Git 状态 |
| --- | --- | --- | --- |
| `project-data/raw/raf-db-basic/` | RAF-DB Basic v1.1 原始标签与图像 | 图像情感模型训练 | 本地保留，不提交 |
| `project-data/processed/raf-db-basic/` | RAF-DB 对齐和预处理结果 | SE-ResNet18 训练与评估 | 本地保留，不提交 |
| `project-data/raw/chinese_emotions.csv` | 中文七分类文本样本 | 文本数据准备 | 提交 |
| `GoEmotions-pytorch/ekman/` | GoEmotions 的 Ekman 映射版本 | 英文文本数据准备 | 提交 |
| `OCEMOTION/` | OCEMOTION 中文情感数据 | 中文文本扩展数据 | 本地保留，不提交 |
| `TESS/` | Toronto Emotional Speech Set | 语音模型训练与评估 | 本地保留，不提交 |
| `CREMA-D/` | Crowd-sourced Emotional Multimodal Actors Dataset | 跨说话人语音评估 | 本地保留，不提交 |
| `EmoDB/` | Berlin Database of Emotional Speech | 跨语料语音评估 | 本地保留，不提交 |

## 目录约定

```text
datasets/
├── project-data/
│   ├── raw/
│   └── processed/
├── GoEmotions-pytorch/
├── OCEMOTION/
├── TESS/
├── CREMA-D/
└── EmoDB/
```

训练脚本已经使用上述路径作为默认值。请分别遵守各数据集的原始许可与引用要求；本仓库的 MIT License 不覆盖第三方数据集。RAF-DB 仅限其许可允许的非商业科研与教学用途。

若只运行已训练模型，无需保留训练数据；本地 `models/` 中的权重和运行依赖仍需存在。