# 中英双语情绪识别模型训练教程

本教程用于两位组员在不同电脑上并行训练模型：

- 组员 A：训练 XLM-R（`xlm-roberta-base`）
- 组员 B：训练 mBERT（`bert-base-multilingual-cased`）

两人必须使用同一个 `dataset_v1`、相同随机种子和相同有效批量大小。禁止各自在本机重新划分数据。

## 1. 训练电脑要求

建议配置：

- Windows 10/11 64位
- NVIDIA独立显卡，建议显存不低于6GB
- 内存建议16GB以上
- 至少15GB可用磁盘空间
- Python 3.11 64位
- 能访问PyTorch和Hugging Face模型下载地址

训练期间连接电源，将Windows电源模式设置为“最佳性能”，暂时关闭自动睡眠。

## 2. 接收项目与统一数据

两位组员都接收完整项目，但可排除以下本机生成目录：

```text
.venv
build
dist
cache
outputs
```

必须包含：

```text
emotion_app/
scripts/
datasets/project-data/processed/dataset_v1/
requirements.txt
requirements-train.txt
```

统一数据目录应包含：

```text
datasets/project-data/processed/dataset_v1/
├── train.csv
├── validation.csv
├── test.csv
└── manifest.json
```

当前正式数据统计：

- 总计：85,153条
- 英文：49,469条
- 中文：35,684条
- 训练集：68,102条
- 验证集：8,514条
- 测试集：8,537条

## 3. 校验数据版本

两人进入项目根目录后都必须执行：

```powershell
Get-FileHash datasets\project-data\processed\dataset_v1\train.csv -Algorithm SHA256
Get-FileHash datasets\project-data\processed\dataset_v1\validation.csv -Algorithm SHA256
Get-FileHash datasets\project-data\processed\dataset_v1\test.csv -Algorithm SHA256
```

结果必须分别为：

```text
train.csv       9AB09538516B95DA4EA58390FEB9C42FFD6C560468F56C9664CD949BD0E2028
validation.csv  5060DA217DEF09FB6FCC277083EF776D18B9E58A1B4E99CF8EDE1CB5F7B153BD
test.csv        61339E070D23B051FB090717E5E8E7EA0C17C5BD22710CFF1AEDA1D82DA00BA0
```

任何一个值不一致都不要训练，应重新复制整个 `dataset_v1`。

## 4. 创建Python环境

在项目根目录打开PowerShell：

```powershell
python --version
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

检查虚拟环境解释器：

```powershell
.venv\Scripts\python -c "import sys; print(sys.executable); print(sys.version)"
```

输出路径应位于当前项目的 `.venv` 中。

## 5. 检查NVIDIA驱动

```powershell
nvidia-smi
```

如果无法识别命令或看不到NVIDIA显卡，应先安装或更新NVIDIA驱动。通常不需要单独安装完整CUDA Toolkit，PyTorch CUDA包会携带训练所需运行库。

## 6. 安装CUDA版PyTorch

先清除可能存在的CPU版：

```powershell
.venv\Scripts\python -m pip uninstall -y torch
```

RTX 30/40系列可先安装CUDA 12.8版本：

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

如果该地址已变更，使用PyTorch官方安装选择器获取当前Windows + Pip + Python + CUDA命令：

https://pytorch.org/get-started/locally/

随后安装项目训练依赖：

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt
```

验证CUDA：

```powershell
.venv\Scripts\python -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('torch_cuda=',torch.version.cuda); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

必须满足：

- `cuda=True`
- PyTorch版本不是`+cpu`
- GPU名称为本机NVIDIA显卡

若为`False`，不得开始正式训练。

## 7. 配置Hugging Face缓存

每次新开PowerShell训练窗口后执行：

```powershell
New-Item -ItemType Directory -Force cache\hf | Out-Null
$env:HF_HOME="$PWD\cache\hf"
$env:HF_HUB_DISABLE_XET="1"
```

首次训练会下载基础模型。Windows出现“不支持符号链接”的警告通常不影响训练。

## 8. 训练前记录环境

```powershell
New-Item -ItemType Directory -Force handoff | Out-Null
nvidia-smi | Out-File handoff\gpu-info.txt -Encoding utf8
.venv\Scripts\python -m pip freeze | Out-File handoff\pip-freeze.txt -Encoding utf8
Copy-Item datasets\project-data\processed\dataset_v1\manifest.json handoff\dataset-manifest.json -Force
```

## 9. 组员A：训练XLM-R

组员A执行：

```powershell
New-Item -ItemType Directory -Force outputs\xlm-roberta | Out-Null

.venv\Scripts\python scripts\train_text_model.py `
  --data-dir datasets\project-data\processed\dataset_v1 `
  --model-name xlm-roberta-base `
  --output-dir outputs\xlm-roberta `
  --epochs 4 `
  --batch-size 4 `
  --gradient-accumulation 4 `
  --learning-rate 2e-5 `
  --max-length 128 `
  --seed 42 2>&1 | Tee-Object outputs\xlm-roberta\train.log
```

有效批量大小为 `4 × 4 = 16`。

如果8GB显存仍然不足，改成：

```text
--batch-size 2
--gradient-accumulation 8
```

有效批量大小仍为16。发生任何参数调整都必须写入交付说明。

## 10. 组员B：训练mBERT

组员B执行：

```powershell
New-Item -ItemType Directory -Force outputs\mbert | Out-Null

.venv\Scripts\python scripts\train_text_model.py `
  --data-dir datasets\project-data\processed\dataset_v1 `
  --model-name bert-base-multilingual-cased `
  --output-dir outputs\mbert `
  --epochs 4 `
  --batch-size 8 `
  --gradient-accumulation 2 `
  --learning-rate 2e-5 `
  --max-length 128 `
  --seed 42 2>&1 | Tee-Object outputs\mbert\train.log
```

有效批量大小为 `8 × 2 = 16`。

如果显存不足，改成：

```text
--batch-size 4
--gradient-accumulation 4
```

## 11. 训练期间监控

另开一个PowerShell窗口：

```powershell
nvidia-smi -l 2
```

正常表现：

- Python进程占用GPU显存
- GPU利用率明显上升
- 每轮结束后终端打印验证集损失、准确率和Macro-F1
- 验证集Macro-F1提高时覆盖保存最优模型

不要在训练期间运行大型游戏、视频渲染或其他GPU程序。

## 12. 训练成功的文件

输出目录至少应包含：

```text
config.json
model.safetensors 或 pytorch_model.bin
tokenizer_config.json
tokenizer.json、vocab.txt 或 sentencepiece.bpe.model等分词器文件
metrics.json
confusion_matrix.csv
train.log
```

检查：

```powershell
Get-ChildItem outputs\xlm-roberta   # 组员A
Get-ChildItem outputs\mbert        # 组员B
```

查看指标：

```powershell
Get-Content -Raw outputs\xlm-roberta\metrics.json   # 组员A
Get-Content -Raw outputs\mbert\metrics.json         # 组员B
```

`metrics.json`应包含：

- `best_validation_macro_f1`
- `test.accuracy`
- `test.macro_f1`
- `test_en.macro_f1`
- `test_zh.macro_f1`
- 每个类别的precision、recall和F1

## 13. 本机推理验收

组员A：

```powershell
.venv\Scripts\python scripts\predict_text.py "今天终于完成项目了，我非常开心" --model outputs\xlm-roberta
.venv\Scripts\python scripts\predict_text.py "I am very happy about this result" --model outputs\xlm-roberta
```

组员B：

```powershell
.venv\Scripts\python scripts\predict_text.py "今天终于完成项目了，我非常开心" --model outputs\mbert
.venv\Scripts\python scripts\predict_text.py "I am very happy about this result" --model outputs\mbert
```

两条命令都应返回：

- `ok: true`
- `error: null`
- 完整七类概率

## 14. 组员A需要提交什么

组员A创建交付目录：

```powershell
New-Item -ItemType Directory -Force handoff\member-a-xlm-roberta | Out-Null
Copy-Item outputs\xlm-roberta\* handoff\member-a-xlm-roberta -Recurse -Force
Copy-Item handoff\gpu-info.txt handoff\member-a-xlm-roberta -Force
Copy-Item handoff\pip-freeze.txt handoff\member-a-xlm-roberta -Force
Copy-Item handoff\dataset-manifest.json handoff\member-a-xlm-roberta -Force
```

另建 `handoff/member-a-xlm-roberta/TRAINING_NOTE.txt`，写明：

```text
成员：A
模型：xlm-roberta-base
实际训练命令：完整复制
开始和结束时间：
是否发生显存不足：
是否修改参数：
异常或中断记录：
```

压缩：

```powershell
Compress-Archive -Path handoff\member-a-xlm-roberta\* -DestinationPath handoff\member-a-xlm-roberta.zip -Force
```

最终提交 `member-a-xlm-roberta.zip`。

## 15. 组员B需要提交什么

组员B创建交付目录：

```powershell
New-Item -ItemType Directory -Force handoff\member-b-mbert | Out-Null
Copy-Item outputs\mbert\* handoff\member-b-mbert -Recurse -Force
Copy-Item handoff\gpu-info.txt handoff\member-b-mbert -Force
Copy-Item handoff\pip-freeze.txt handoff\member-b-mbert -Force
Copy-Item handoff\dataset-manifest.json handoff\member-b-mbert -Force
```

另建 `handoff/member-b-mbert/TRAINING_NOTE.txt`：

```text
成员：B
模型：bert-base-multilingual-cased
实际训练命令：完整复制
开始和结束时间：
是否发生显存不足：
是否修改参数：
异常或中断记录：
```

压缩：

```powershell
Compress-Archive -Path handoff\member-b-mbert\* -DestinationPath handoff\member-b-mbert.zip -Force
```

最终提交 `member-b-mbert.zip`。

## 16. 负责人收到文件后的检查

负责人应收到：

```text
member-a-xlm-roberta.zip
member-b-mbert.zip
```

每个压缩包必须同时包含：

1. 完整模型权重和分词器
2. `metrics.json`
3. `confusion_matrix.csv`
4. `train.log`
5. `gpu-info.txt`
6. `pip-freeze.txt`
7. `dataset-manifest.json`
8. `TRAINING_NOTE.txt`

缺少模型权重、分词器或`config.json`时，模型无法接入应用；只提交`metrics.json`是不够的。

## 17. 常见错误

### `torch.cuda.is_available()`为False

- 确认使用的是项目 `.venv` 中的Python
- 确认PyTorch版本不带 `+cpu`
- 重新安装CUDA版PyTorch
- 更新NVIDIA驱动后重启

### CUDA out of memory

- 减小 `--batch-size`
- 同比例增大 `--gradient-accumulation`
- 关闭其他占用GPU的软件
- 不要修改 `max-length`、数据或随机种子来偷偷规避问题

### Hugging Face下载权限错误

重新执行：

```powershell
$env:HF_HOME="$PWD\cache\hf"
$env:HF_HUB_DISABLE_XET="1"
```

### 下载失败或超时

不要删除已经下载的 `cache/hf`。网络恢复后重新执行相同训练命令，Hugging Face会复用已下载文件；训练本身不支持自动从中断轮次恢复，因此进入正式训练后应尽量避免关机。

### 两位成员指标无法公平比较

必须核对：

- 三个数据文件SHA-256
- 随机种子42
- 训练4轮
- 学习率`2e-5`
- 最大长度128
- 有效批量大小16

