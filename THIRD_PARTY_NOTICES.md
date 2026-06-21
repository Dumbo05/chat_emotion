# 第三方开源声明

本项目基于下列开源工作进行改造。上游源码保存在 `vendor/`，便于复现、审计和履行许可证义务。

## Sentiment-analysis

- 来源：https://github.com/hchhtc123/Sentiment-analysis
- 许可证：MIT
- 使用范围：参考并改造 PyQt5 单条/批量文本分类交互。
- 主要修改：替换 PaddleHub 推理，改为三模态标签页、异步任务和七类概率展示。
- 许可证原文：`vendor/sentiment-analysis/LICENSE`

## GoEmotions-pytorch

- 来源：https://github.com/monologg/GoEmotions-pytorch
- 许可证：Apache License 2.0
- 使用范围：Ekman 七类标签、固定数据划分和训练流程参考。
- 主要修改：将英文 BERT 多标签流程改造为中英双语单标签 mBERT/XLM-R 对比流程；多标签样本明确排除。
- 许可证原文：`vendor/goemotions-pytorch/LICENSE`

## multimodal-emotion-classification

- 来源：https://github.com/RachaCodez/multimodal-emotion-classification
- 许可证：MIT
- 使用范围：七类统一结果结构，以及语音、图像预处理和推理扩展点参考。
- 主要修改：不使用 Flask、数据库和启发式回退；一期只保留本地 PyQt5 扩展接口。
- 许可证原文：`vendor/multimodal-emotion-classification/LICENSE`

## GoEmotions 数据

GoEmotions 数据集来自 Google Research，数据按 CC BY 4.0 提供。论文及数据来源必须在实验报告中引用。当前仓库中的 Ekman TSV 副本来自上述 `GoEmotions-pytorch` 上游。


## Image recognition models

Image and camera recognition use OpenCV Zoo's YuNet face detector (MIT license). The expression classifier is this project's SE-ResNet18, trained from random initialization on RAF-DB Basic; it does not use the OpenCV Zoo expression model. RAF-DB images are restricted to non-commercial research and educational use. Sources: https://github.com/opencv/opencv_zoo and http://whdeng.cn/RAF/model1.html
