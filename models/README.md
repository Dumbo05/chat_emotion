# 模型目录

模型权重通常体积较大，且部分权重受上游模型或数据许可约束，因此 Git 仅跟踪本说明文件。本地模型不会在工程清理时删除。

## 运行时布局

```text
models/
├── text/                          # Hugging Face 七分类文本模型
├── image/
│   └── rafdb_se_resnet18/
│       └── rafdb_emotion.onnx
└── speech/
    ├── wavlm_simsan_encoder.onnx
    ├── wavlm_simsan_head.joblib
    └── speech_model.joblib        # 可选的传统特征回退模型
```

文本模型的 `config.json` 应包含完整的 `id2label`，标签为 `anger`、`disgust`、`fear`、`joy`、`sadness`、`surprise` 和 `neutral`。也可用环境变量 `EMOTION_TEXT_MODEL` 指向其他本地 Hugging Face 模型目录。

图像模型由 `scripts/train_rafdb_model.py` 在 RAF-DB Basic v1.1 上从随机初始化训练，推理时使用水平翻转测试时增强（Flip TTA）。官方测试集准确率为 77.71%，Macro-F1 为 69.22%；Flip TTA 准确率为 78.16%。

语音主模型采用 WavLM 表征与 SIMSAN 分类头，ONNX 编码器和 Joblib 分类头共同构成运行时检查点。传统 MFCC/RBF-SVM 模型仅用于兼容和回退。

模型权重、训练缓存和中间检查点不会提交到 Git。复现实验时请按 README 和训练脚本生成，并遵守上游模型及数据集许可。