# 中英双语多模态情绪识别系统

基于 PyQt5 的本地桌面应用。提供真实的中英双语文本与 TESS 语音情感识别，图像页面保留统一扩展接口。系统固定输出愤怒、厌恶、恐惧、喜悦、悲伤、惊讶、中性七类。

## 已实现功能

- 中文与英文单条文本识别、七类概率和置信度展示
- Excel 批量识别，自动识别 `text/content/文本/内容` 列
- 模型加载和推理运行于 `QThread`，界面保持响应
- 语音、图像文件选择与预览占位；未配置模型时禁止预测
- mBERT 与 XLM-R 共用的数据准备、训练、评估和命令行预测脚本
- 固定数据划分、语言分组指标、Macro-F1、分类报告和混淆矩阵
- 模型缺失或损坏时明确报错，不提供随机或关键词伪预测

## 快速启动

推荐 Python 3.10 或 3.11。Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

首次启动前，将训练好的模型放入 `models/text/`。没有模型时应用仍可启动，但文本预测会明确提示模型不可用。

## 数据准备

英文数据已随上游 GoEmotions-pytorch 以 Ekman 七类形式保存在 `vendor/`。准备合法取得的中文情绪 CSV 后执行：

```powershell
.venv\Scripts\python scripts\prepare_dataset.py `
  --goemotions-dir vendor\goemotions-pytorch\data\ekman `
  --chinese-csv data\raw\chinese_emotions.csv `
  --output-dir data\processed\dataset_v1
```

脚本会排除多标签英文样本，而不是任意挑选一个标签。若暂时没有中文数据，可省略 `--chinese-csv` 先验证英文流程。

## 多电脑并行训练

所有成员必须使用同一个 `dataset_v1`，并核对 `manifest.json` 的 SHA-256。两台电脑分别执行：

```powershell
# 3号成员：mBERT
.venv\Scripts\python scripts\train_text_model.py `
  --data-dir data\processed\dataset_v1 `
  --model-name bert-base-multilingual-cased `
  --output-dir outputs\mbert

# 4号成员：XLM-R
.venv\Scripts\python scripts\train_text_model.py `
  --data-dir data\processed\dataset_v1 `
  --model-name xlm-roberta-base `
  --output-dir outputs\xlm-roberta
```

训练脚本按验证集 Macro-F1 保存最优模型，并在测试集输出总体、中文和英文指标。选择获胜模型后，将整个输出目录复制为 `models/text/`。

## 命令行一致性检查

```powershell
.venv\Scripts\python scripts\predict_text.py "今天真是太开心了" --model models\text
```

命令行与界面共用同一个 `TextRecognizer`，因此相同输入和模型应得到相同结果。

## Excel格式

输入文件应包含以下任意一个文本列名：`text`、`content`、`文本`、`内容`。输出增加：

- `预测情绪`
- `置信度`
- `错误信息`

若模型不可用，错误会逐行写入，而不会生成伪预测。

## 测试与打包

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\python -m pytest

.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\pyinstaller emotion_app.spec
```

模型通常较大，默认不嵌入 EXE；发布时将 `models/text/` 与生成的程序放在同一发布目录，并在目标 Windows 电脑进行启动和预测验收。

## 五人协作入口

1. 项目集成：根目录、统一接口、许可证和最终模型合并。
2. 数据处理：`scripts/prepare_dataset.py` 与 `dataset_v1`。
3. mBERT训练：`scripts/train_text_model.py --model-name bert-base-multilingual-cased`。
4. XLM-R训练：同一脚本使用 `xlm-roberta-base`。
5. PyQt5界面：`emotion_app/ui/`、批量识别与打包验收。

第三方来源和修改范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。


## 语音情感识别

语音模块支持 WAV 与 MP3。MP3 会先通过内置 miniaudio 解码为 16 kHz 单声道音频，无需安装 FFmpeg，也不需要重新训练模型。训练阶段使用 TESS 数据集，提取 MFCC、Delta-MFCC、能量、过零率、频谱质心、滚降频率和时长统计特征，再由标准化 RBF-SVM 完成七分类。训练集、验证集和测试集按说出的单词分组为 70%/15%/15%，同一个单词不会跨集合，避免词汇泄漏。

```powershell
.venv\Scripts\python scripts\train_speech_model.py `
  --data-dir archive_sound `
  --output-dir models\speech
```

训练产物包括 `speech_model.joblib`、`metrics.json`、`confusion_matrix.csv` 和可复用的 `features.npz`。当前数据中两条 `.wav` 实为 AIFF，脚本会明确跳过。评估指标保存在 `metrics.json` 中，不在桌面界面展示；界面仅显示单条音频的预测情绪、置信度与七类概率。报告同时包含双向留一说话人评估；当前跨说话人平均准确率为 47.75%、Macro-F1 为 40.16%，说明 TESS 上的同说话人高分不能直接代表真实开放环境泛化能力。


### Windows EXE 交付

`dist/EmotionRecognition.exe` 已内置语音模型、评估指标及 NumPy/SciPy/scikit-learn 推理依赖，可直接进行 WAV 语音情感识别，不需要旁置 `models/speech`。当前仓库没有 `models/text` 权重，因此交付 EXE 未嵌入体积约 4.4 GB 的 PyTorch 文本运行时；源码环境中的文本训练与推理流程不受影响。


## 图像与摄像头情绪识别

图像页支持导入 PNG、JPG、BMP、WebP 文件，也可启动电脑默认摄像头进行实时识别。系统使用 OpenCV Zoo 的 YuNet 检测人脸并以五点关键点对齐，再由 Progressive Teacher / MobileFaceNet 模型输出七类表情概率。模型随 EXE 内置，推理完全在本地进行。摄像头无法打开时，请检查 Windows 相机隐私权限以及摄像头是否被其他程序占用。
