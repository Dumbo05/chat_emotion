# 模型目录

将训练完成的 Hugging Face 七分类模型完整复制到 `models/text/`。目录至少应包含：

- `config.json`
- 模型权重（`model.safetensors` 或 `pytorch_model.bin`）
- 分词器配置和词表文件

`config.json` 的 `id2label` 必须完整包含：`anger`、`disgust`、`fear`、`joy`、`sadness`、`surprise`、`neutral`。

模型文件未提交到 Git。也可通过环境变量 `EMOTION_TEXT_MODEL` 指向其他本地模型目录。

