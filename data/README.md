# 数据目录

不要把未经授权的数据集直接提交到仓库。使用数据准备脚本生成团队共享的固定版本：

```powershell
.venv\Scripts\python scripts\prepare_dataset.py `
  --goemotions-dir vendor\goemotions-pytorch\data\ekman `
  --chinese-csv data\raw\chinese_emotions.csv `
  --output-dir data\processed\dataset_v1
```

中文 CSV 接受 `text/content/文本/内容` 文本列和 `label/emotion/情绪/标签` 标签列。生成结果包含固定的训练、验证、测试 CSV 和带 SHA-256 校验值的 `manifest.json`。

