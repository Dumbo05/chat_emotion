# 文本情绪识别 24GB GPU 服务器 24 小时冲刺教程

生成日期：2026-06-24

本文档用于把一台 24GB NVIDIA GPU 服务器用于文本情绪识别模型冲刺。当前本地最佳文本结果约为：

| 模型 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| XLM-R base 最佳实验 | 0.6812 | 0.6135 |
| mBERT 最佳 sweep | 0.6694 | 0.6019 |

目标是尽量接近 80% Accuracy。这个目标不能只靠换一个大模型，必须同时管理数据版本、验证集选型、弱类别、日志和后续集成。

## 1. 核心原则

1. 不要用 test 反复调参。当前脚本每次训练结束都会输出 test 指标，短期冲刺可以看作参考，但正式论文/答辩应以 validation 选型，只在最终候选上汇报 test。
2. 24 小时优先跑最可能提升的模型：`xlm-roberta-large`、`mdeberta-v3-base`，再补 XLM-R large 多 seed。
3. 如果 `xlm-roberta-large` 第一轮冲不到 72%，先怀疑数据/标签/类别不均衡，而不是继续盲目堆更大模型。
4. 如果 `xlm-roberta-large` 冲到 75% 左右，下一步优先做多 seed ensemble、class weight/focal loss、弱类增强，而不是马上换路线。

## 2. 服务器建议配置

最低建议：

| 项目 | 建议 |
|---|---|
| GPU | RTX 3090 / RTX 4090 / A5000 / A10 / A6000，24GB 显存 |
| CPU | 8 核以上 |
| 内存 | 32GB 以上 |
| 磁盘 | 至少 80GB 空闲，推荐 SSD |
| 系统 | Ubuntu 20.04/22.04 |
| Python | 3.10 或 3.11 |

24GB 显存下，`xlm-roberta-large` 建议从 `batch-size=2, gradient-accumulation=8` 开始，有余量再改为 `batch-size=4, gradient-accumulation=4`。

## 3. 需要准备的数据集

### 3.1 服务器训练必须传的本地数据

只训练模型时，不需要重新下载原始数据，直接传 processed 数据即可：

```text
C:\Users\Lenovo\Desktop\del\datasets\project-data\processed\dataset_v1
```

服务器上建议放到：

```text
~/emotion-text/datasets/project-data/processed/dataset_v1
```

该目录必须包含：

```text
manifest.json
train.csv
validation.csv
test.csv
```

当前 `dataset_v1` 统计：

| 项目 | 数值 |
|---|---:|
| 总样本 | 85153 |
| train | 68102 |
| validation | 8514 |
| test | 8537 |
| 英文 | 49469 |
| 中文 | 35684 |

标签分布：

| 标签 | 数量 |
|---|---:|
| joy | 32079 |
| neutral | 16021 |
| sadness | 15479 |
| anger | 9535 |
| surprise | 5713 |
| disgust | 5039 |
| fear | 1287 |

明显问题：`fear` 很少，`joy` 很多。Accuracy 容易被大类拉高，Macro-F1 才是更稳的指标。

### 3.2 校验哈希

服务器上传完后，必须校验：

```bash
cd ~/emotion-text

sha256sum datasets/project-data/processed/dataset_v1/train.csv
sha256sum datasets/project-data/processed/dataset_v1/validation.csv
sha256sum datasets/project-data/processed/dataset_v1/test.csv
```

必须匹配：

```text
train.csv       9ab09538516b95da4ea58390feb9c42fffd6c560468f56c9664cd949bd0e2028
validation.csv  5060da217def09fb6fcc277083ef776d18b9e58a1b4e99cf8ede1cb5f7b153bd
test.csv        61339e070d23b051fb090717e5e8e7ea0c17c5bd22710cff1aeda1d82da00ba0
```

如果不一致，停止训练，重新上传数据。

### 3.3 可选英文子集

英文单独诊断可传：

```text
C:\Users\Lenovo\Desktop\del\datasets\project-data\processed\dataset_v1_english
```

服务器路径：

```text
~/emotion-text/datasets/project-data/processed/dataset_v1_english
```

用途：判断模型在英文 GoEmotions 派生数据上的上限。如果英文单独能高很多，而双语数据低，说明中文数据/跨语种分布是瓶颈。

### 3.4 如果需要重建数据才需要的原始数据

本轮服务器训练不建议重建数据。若必须重建，原始来源如下：

| 数据 | 来源/网址 | 本地路径 |
|---|---|---|
| GoEmotions 官方数据 | https://huggingface.co/datasets/google-research-datasets/go_emotions | 通过脚本派生到 `dataset_v1` |
| Google Research GoEmotions 仓库 | https://github.com/google-research/google-research/tree/master/goemotions | 参考来源 |
| 项目当前中文情绪数据 | 无公开远程地址，使用本地文件 | `C:\Users\Lenovo\Desktop\del\datasets\project-data\raw\chinese_emotions.csv` |
| 中文数据来源说明 | 本地说明文件 | `C:\Users\Lenovo\Desktop\del\datasets\project-data\raw\OCEMOTION_SOURCE.md` |

项目当前正式训练使用的是已处理好的 `dataset_v1`，来源记录在：

```text
C:\Users\Lenovo\Desktop\del\datasets\project-data\processed\dataset_v1\manifest.json
```

## 4. 模型候选和网址

优先级从高到低：

| 优先级 | 模型 | Hugging Face 名称 | 网址 | 用途 |
|---:|---|---|---|---|
| 1 | XLM-R large | `FacebookAI/xlm-roberta-large` | https://huggingface.co/FacebookAI/xlm-roberta-large | 双语主力模型 |
| 2 | mDeBERTa v3 base | `microsoft/mdeberta-v3-base` | https://huggingface.co/microsoft/mdeberta-v3-base | 双语/多语替代路线 |
| 3 | Chinese RoBERTa WWM large | `hfl/chinese-roberta-wwm-ext-large` | https://huggingface.co/hfl/chinese-roberta-wwm-ext-large | 中文分项诊断，不适合作为双语唯一模型 |
| 4 | XLM-R base 复现实验 | `xlm-roberta-base` 或 `FacebookAI/xlm-roberta-base` | https://huggingface.co/FacebookAI/xlm-roberta-base | 低成本 sanity check |

说明：

- `xlm-roberta-large` 是本轮 24GB GPU 的第一主线。
- `mdeberta-v3-base` 参数量不是最大，但经常在多语分类上比旧 mBERT/XLM-R base 更强，值得跑。
- `hfl/chinese-roberta-wwm-ext-large` 只能作为中文诊断模型。当前训练 CSV 是中英混合，如果直接用它训练全量双语数据，英文会明显吃亏。

## 5. 服务器环境配置

### 5.1 建项目目录

```bash
mkdir -p ~/emotion-text
cd ~/emotion-text
```

把本地项目中的这些目录/文件传到服务器：

```text
emotion_app/
scripts/
datasets/project-data/processed/dataset_v1/
requirements.txt
requirements-train.txt
```

可以用 `scp`，示例：

```bash
scp -r emotion_app scripts datasets requirements.txt requirements-train.txt USER@SERVER_IP:~/emotion-text/
```

### 5.2 安装环境

```bash
cd ~/emotion-text

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

如果服务器 CUDA 版本或驱动不适配 CUDA 12.8，按 PyTorch 官网当前命令替换：

```text
https://pytorch.org/get-started/locally/
```

### 5.3 验证 CUDA

```bash
source .venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

必须看到：

```text
cuda: True
```

### 5.4 Hugging Face 缓存

```bash
cd ~/emotion-text
mkdir -p cache/hf
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1
```

如果模型下载失败，先单独测试：

```bash
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
print("download ok")
PY
```

## 6. 推荐 24 小时实验排程

### 阶段 A：第 0-1 小时，环境和数据验收

执行：

```bash
cd ~/emotion-text
source .venv/bin/activate

nvidia-smi
sha256sum datasets/project-data/processed/dataset_v1/train.csv
sha256sum datasets/project-data/processed/dataset_v1/validation.csv
sha256sum datasets/project-data/processed/dataset_v1/test.csv
```

创建日志目录：

```bash
mkdir -p outputs/text logs handoff
nvidia-smi > handoff/gpu-info.txt
pip freeze > handoff/pip-freeze.txt
cp datasets/project-data/processed/dataset_v1/manifest.json handoff/dataset-manifest.json
```

### 阶段 B：第 1-8 小时，XLM-R large 基线

先跑最重要的一轮：

```bash
cd ~/emotion-text
source .venv/bin/activate
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1

RUN=xlmr_large_lr1e-5_len128_seed42
mkdir -p outputs/text/$RUN

python scripts/train_text_model.py \
  --data-dir datasets/project-data/processed/dataset_v1 \
  --model-name FacebookAI/xlm-roberta-large \
  --output-dir outputs/text/$RUN \
  --epochs 4 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --learning-rate 1e-5 \
  --max-length 128 \
  --seed 42 2>&1 | tee outputs/text/$RUN/train.log
```

如果 OOM，把 batch 调低：

```text
--batch-size 1
--gradient-accumulation 16
```

如果显存低于 20GB 且 GPU 利用率不满，可以改：

```text
--batch-size 4
--gradient-accumulation 4
```

### 阶段 C：第 8-20 小时，按结果分支执行

先快速汇总第一轮：

```bash
python - <<'PY'
import json
from pathlib import Path

for p in sorted(Path("outputs/text").glob("*/metrics.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    print(p.parent.name)
    print("  model:", m.get("model_name"))
    print("  best_val_macro_f1:", round(m.get("best_validation_macro_f1", 0), 6))
    print("  test_acc:", round(m["test"]["accuracy"], 6))
    print("  test_macro_f1:", round(m["test"]["macro_f1"], 6))
    for k in ("test_en", "test_zh"):
        if k in m:
            print(f"  {k}_acc:", round(m[k]["accuracy"], 6), f"{k}_macro_f1:", round(m[k]["macro_f1"], 6))
PY
```

#### 分支 1：如果 XLM-R large Accuracy >= 0.75

说明 large 路线有效。接下来不要换模型，先榨干 XLM-R large：

```bash
for LR in 1e-5 2e-5; do
  for SEED in 123 456; do
    RUN=xlmr_large_lr${LR}_len128_seed${SEED}
    mkdir -p outputs/text/$RUN
    python scripts/train_text_model.py \
      --data-dir datasets/project-data/processed/dataset_v1 \
      --model-name FacebookAI/xlm-roberta-large \
      --output-dir outputs/text/$RUN \
      --epochs 4 \
      --batch-size 2 \
      --gradient-accumulation 8 \
      --learning-rate $LR \
      --max-length 128 \
      --seed $SEED 2>&1 | tee outputs/text/$RUN/train.log
  done
done
```

之后行动：

1. 选 validation Macro-F1 最高的 3 个模型。
2. 做 soft-voting ensemble。
3. 如果 ensemble 后接近 78%，再加入 class weight/focal loss 版本。
4. 如果单模型已经 78%-80%，冻结配置，停止继续看 test，准备最终报告。

#### 分支 2：如果 XLM-R large Accuracy 在 0.72-0.75

说明 large 有提升但不够。先补学习率和长度：

```bash
for CFG in \
  "1e-5 192 42" \
  "2e-5 128 42" \
  "2e-5 192 42" \
  "8e-6 160 123"
do
  set -- $CFG
  LR=$1
  LEN=$2
  SEED=$3
  RUN=xlmr_large_lr${LR}_len${LEN}_seed${SEED}
  mkdir -p outputs/text/$RUN
  python scripts/train_text_model.py \
    --data-dir datasets/project-data/processed/dataset_v1 \
    --model-name FacebookAI/xlm-roberta-large \
    --output-dir outputs/text/$RUN \
    --epochs 4 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --learning-rate $LR \
    --max-length $LEN \
    --seed $SEED 2>&1 | tee outputs/text/$RUN/train.log
done
```

之后行动：

1. 看 `test_zh` 和 `test_en` 哪个拖后腿。
2. 如果中文拖后腿，跑中文专用模型做诊断。
3. 如果弱类 `fear/disgust/anger` 拖后腿，下一轮改训练脚本加入 class weight 或 focal loss。

#### 分支 3：如果 XLM-R large Accuracy < 0.72

说明大模型本身没有解决核心问题。立刻换模型跑 mDeBERTa：

```bash
for LR in 1e-5 2e-5; do
  RUN=mdeberta_v3_base_lr${LR}_len128_seed42
  mkdir -p outputs/text/$RUN
  python scripts/train_text_model.py \
    --data-dir datasets/project-data/processed/dataset_v1 \
    --model-name microsoft/mdeberta-v3-base \
    --output-dir outputs/text/$RUN \
    --epochs 4 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate $LR \
    --max-length 128 \
    --seed 42 2>&1 | tee outputs/text/$RUN/train.log
done
```

如果 mDeBERTa 也不到 72%，继续堆模型的收益会很差。下一步应转向：

1. 数据清洗：抽查高置信错例。
2. 重新处理多标签/弱标签样本。
3. 对 `fear/disgust/anger` 做训练集增强。
4. 训练脚本加入 class weight/focal loss。

### 阶段 D：第 20-24 小时，补诊断和保存结果

跑中文专用模型诊断中文分项。注意：它不是最终双语模型，只用于判断中文数据上限。

```bash
RUN=chinese_roberta_wwm_large_full_bilingual_diag
mkdir -p outputs/text/$RUN

python scripts/train_text_model.py \
  --data-dir datasets/project-data/processed/dataset_v1 \
  --model-name hfl/chinese-roberta-wwm-ext-large \
  --output-dir outputs/text/$RUN \
  --epochs 4 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 2e-5 \
  --max-length 128 \
  --seed 42 2>&1 | tee outputs/text/$RUN/train.log
```

如果你上传了 `dataset_v1_english`，也可以跑英文子集 sanity check：

```bash
RUN=xlmr_large_english_only_diag
mkdir -p outputs/text/$RUN

python scripts/train_text_model.py \
  --data-dir datasets/project-data/processed/dataset_v1_english \
  --model-name FacebookAI/xlm-roberta-large \
  --output-dir outputs/text/$RUN \
  --epochs 4 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --learning-rate 1e-5 \
  --max-length 128 \
  --seed 42 2>&1 | tee outputs/text/$RUN/train.log
```

## 7. 一键顺序跑脚本

如果你想让服务器自己跑满，可以保存下面内容为 `run_text_24h.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/emotion-text
source .venv/bin/activate
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1
mkdir -p outputs/text

run_one() {
  local run="$1"
  local model="$2"
  local lr="$3"
  local len="$4"
  local seed="$5"
  local bs="$6"
  local ga="$7"

  if [ -f "outputs/text/$run/metrics.json" ]; then
    echo "skip existing $run"
    return
  fi

  mkdir -p "outputs/text/$run"
  echo "===== START $run ====="
  date
  nvidia-smi

  python scripts/train_text_model.py \
    --data-dir datasets/project-data/processed/dataset_v1 \
    --model-name "$model" \
    --output-dir "outputs/text/$run" \
    --epochs 4 \
    --batch-size "$bs" \
    --gradient-accumulation "$ga" \
    --learning-rate "$lr" \
    --max-length "$len" \
    --seed "$seed" 2>&1 | tee "outputs/text/$run/train.log"

  echo "===== END $run ====="
  date
}

run_one xlmr_large_lr1e-5_len128_seed42 FacebookAI/xlm-roberta-large 1e-5 128 42 2 8
run_one xlmr_large_lr2e-5_len128_seed42 FacebookAI/xlm-roberta-large 2e-5 128 42 2 8
run_one xlmr_large_lr1e-5_len192_seed42 FacebookAI/xlm-roberta-large 1e-5 192 42 2 8
run_one xlmr_large_lr1e-5_len128_seed123 FacebookAI/xlm-roberta-large 1e-5 128 123 2 8
run_one mdeberta_v3_base_lr1e-5_len128_seed42 microsoft/mdeberta-v3-base 1e-5 128 42 4 4
run_one mdeberta_v3_base_lr2e-5_len128_seed42 microsoft/mdeberta-v3-base 2e-5 128 42 4 4

python - <<'PY'
import json
from pathlib import Path

rows = []
for p in sorted(Path("outputs/text").glob("*/metrics.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "run": p.parent.name,
        "model": m.get("model_name"),
        "val_macro_f1": m.get("best_validation_macro_f1"),
        "test_acc": m["test"]["accuracy"],
        "test_macro_f1": m["test"]["macro_f1"],
        "en_macro_f1": m.get("test_en", {}).get("macro_f1"),
        "zh_macro_f1": m.get("test_zh", {}).get("macro_f1"),
    })

rows.sort(key=lambda r: (r["val_macro_f1"] or 0), reverse=True)
print("| rank | run | model | val_macro_f1 | test_acc | test_macro_f1 | en_macro_f1 | zh_macro_f1 |")
print("|---:|---|---|---:|---:|---:|---:|---:|")
for i, r in enumerate(rows, 1):
    print("| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {} | {} |".format(
        i,
        r["run"],
        r["model"],
        r["val_macro_f1"] or 0,
        r["test_acc"],
        r["test_macro_f1"],
        "" if r["en_macro_f1"] is None else f'{r["en_macro_f1"]:.6f}',
        "" if r["zh_macro_f1"] is None else f'{r["zh_macro_f1"]:.6f}',
    ))
PY
```

运行：

```bash
chmod +x run_text_24h.sh
nohup bash run_text_24h.sh > logs/run_text_24h.nohup.log 2>&1 &
tail -f logs/run_text_24h.nohup.log
```

监控 GPU：

```bash
watch -n 2 nvidia-smi
```

## 8. 如何判断下一步

### 8.1 XLM-R large >= 75%

继续做：

1. XLM-R large 多 seed：`seed=123,456,789`。
2. Top 3 模型 soft-voting ensemble。
3. 加 class weight/focal loss 版本。
4. 对弱类做增强，只增强训练集。

不要马上换掉 XLM-R large。这个分数说明主路线是对的。

### 8.2 XLM-R large 72%-75%

继续做：

1. 跑 `max_length=160/192`。
2. 跑 `lr=8e-6/1e-5/2e-5`。
3. 跑 mDeBERTa 对照。
4. 分析中文/英文分项和逐类 F1。

如果 mDeBERTa 比 XLM-R large 高，后续转 mDeBERTa；否则继续 XLM-R large。

### 8.3 XLM-R large < 72%

不要继续盲目堆 XLM-R large seed。优先：

1. 跑 `microsoft/mdeberta-v3-base`。
2. 抽查 validation/test 高置信错例。
3. 检查 `fear/disgust/anger` 标签质量。
4. 做 class weight/focal loss。
5. 重新评估 80% 目标是否需要改变数据构造。

### 8.4 large 和 mDeBERTa 都 < 72%

这基本说明瓶颈不是服务器，而是数据。下一步应该做数据工程：

1. 清理短文本、空文本、标点噪声。
2. 抽查每类 100 条高置信错例。
3. 检查中英文标签映射是否一致。
4. 对 `fear` 进行扩充或重采样。
5. 使用两阶段分类：先正/负/中性，再细分七类。

## 9. 结果回传

训练完先压缩这些：

```bash
cd ~/emotion-text
tar -czf text_runs_$(date +%Y%m%d_%H%M).tar.gz outputs/text handoff logs
```

只回传最佳模型时，至少拿回：

```text
outputs/text/<best_run>/config.json
outputs/text/<best_run>/model.safetensors
outputs/text/<best_run>/tokenizer.json
outputs/text/<best_run>/tokenizer_config.json
outputs/text/<best_run>/metrics.json
outputs/text/<best_run>/confusion_matrix.csv
outputs/text/<best_run>/train.log
```

## 10. 最终汇报模板

训练结束后按这个表汇报：

| rank | run | model | val_macro_f1 | test_acc | test_macro_f1 | en_macro_f1 | zh_macro_f1 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 |  |  |  |  |  |  |  |

然后写一句结论：

```text
当前最佳为 <run>，validation Macro-F1=<...>，test Accuracy=<...>，test Macro-F1=<...>。
相比本地 XLM-R base 最佳 Accuracy +<...> pp，Macro-F1 +<...> pp。
```

## 11. 重要提醒

1. 80% Accuracy 是高目标。如果 XLM-R large 只能到 72%-75%，不是失败，而是说明当前数据集上限可能受标签噪声和类别不均衡限制。
2. `fear` 只有 1287 条，是最需要处理的弱类。
3. 所有增强、重采样、清洗都只能作用于 train，不能动 validation/test。
4. 如果要写论文，最终必须讲清楚：数据来源、标签映射、划分哈希、模型选择依据、是否使用 test 调参。

## 12. 你的 RTX 3090 服务器专用配置说明

你的机器配置可以直接用于本项目文本大模型训练：

| 项目 | 当前配置 | 训练判断 |
|---|---|---|
| CPU | Intel Core i9-10940X，14 核 28 线程，最高 4.8GHz | 足够，数据加载不会成为主要瓶颈 |
| GPU | NVIDIA GeForce RTX 3090 | 合适，24GB 显存可跑 `xlm-roberta-large` |
| 显存 | 24GB | 推荐 `batch-size=2, gradient-accumulation=8` 起步 |
| 内存 | 约 128GB | 很宽裕 |
| 硬盘 | 1TB 系统盘 + 2TB 数据盘 | 足够放模型缓存、数据和多轮实验输出 |
| 系统 | Ubuntu 26.04 LTS | 可以用，但系统偏新，优先使用 Python venv 隔离环境 |
| NVIDIA 驱动 | 595.71.05 | 足够新 |
| `nvidia-smi` CUDA Version | 13.2 | 这是驱动支持的 CUDA 上限，不要求安装 CUDA 13.2 Toolkit |
| 网络 | Tailscale + SSH | 很适合远程长时间训练 |

重点：PyTorch 的 pip wheel 自带 CUDA runtime。只要驱动足够新，安装 `cu128` 通常就能正常使用 RTX 3090。不要因为 `nvidia-smi` 显示 CUDA 13.2 就强行找 CUDA 13.2 版 PyTorch。

## 13. 从本机传文件到服务器的详细教程

下面假设：

```text
本机项目路径：C:\Users\Lenovo\Desktop\del
服务器项目路径：~/emotion-text
服务器用户名：ubuntu
服务器地址：100.64.12.34
SSH 端口：22
```

你需要把 `ubuntu`、`100.64.12.34`、`22` 换成自己的信息。Tailscale 地址一般是 `100.x.y.z`。如果你用的是公网 IP，也可以直接填公网 IP。

### 13.1 先测试 SSH 能否登录

在本机 Windows PowerShell 中执行：

```powershell
ssh ubuntu@100.64.12.34
```

如果 SSH 端口不是 22，例如是 2222：

```powershell
ssh -p 2222 ubuntu@100.64.12.34
```

第一次连接会询问是否信任主机，输入：

```text
yes
```

登录成功后，在服务器上执行：

```bash
pwd
nvidia-smi
exit
```

能看到 RTX 3090，就说明远程通道正常。

### 13.2 服务器先创建项目目录

在本机 PowerShell 执行远程命令：

```powershell
ssh ubuntu@100.64.12.34 "mkdir -p ~/emotion-text"
```

如果端口不是 22：

```powershell
ssh -p 2222 ubuntu@100.64.12.34 "mkdir -p ~/emotion-text"
```

### 13.3 方式 A：直接用 scp 传必要文件，最简单

在本机 PowerShell 执行：

```powershell
cd C:\Users\Lenovo\Desktop\del

scp -r emotion_app scripts requirements.txt requirements-train.txt ubuntu@100.64.12.34:~/emotion-text/
scp -r datasets\project-data\processed\dataset_v1 ubuntu@100.64.12.34:~/emotion-text/datasets/project-data/processed/
```

如果需要英文子集诊断，也传：

```powershell
scp -r datasets\project-data\processed\dataset_v1_english ubuntu@100.64.12.34:~/emotion-text/datasets/project-data/processed/
```

如果 SSH 端口不是 22，`scp` 用大写 `-P`：

```powershell
scp -P 2222 -r emotion_app scripts requirements.txt requirements-train.txt ubuntu@100.64.12.34:~/emotion-text/
scp -P 2222 -r datasets\project-data\processed\dataset_v1 ubuntu@100.64.12.34:~/emotion-text/datasets/project-data/processed/
```

注意：`ssh` 使用小写 `-p`，`scp` 使用大写 `-P`。

### 13.4 方式 B：先压缩再上传，更稳定

如果小文件很多导致 `scp -r` 慢，推荐先打包。

在本机 PowerShell 执行：

```powershell
cd C:\Users\Lenovo\Desktop\del

New-Item -ItemType Directory -Force handoff | Out-Null

tar -czf handoff\emotion_text_code_data.tar.gz `
  emotion_app `
  scripts `
  requirements.txt `
  requirements-train.txt `
  datasets\project-data\processed\dataset_v1
```

上传压缩包：

```powershell
scp handoff\emotion_text_code_data.tar.gz ubuntu@100.64.12.34:~/
```

在服务器解压：

```powershell
ssh ubuntu@100.64.12.34 "mkdir -p ~/emotion-text && tar -xzf ~/emotion_text_code_data.tar.gz -C ~/emotion-text"
```

如果你还要传英文子集，可以单独打包：

```powershell
tar -czf handoff\dataset_v1_english.tar.gz datasets\project-data\processed\dataset_v1_english
scp handoff\dataset_v1_english.tar.gz ubuntu@100.64.12.34:~/
ssh ubuntu@100.64.12.34 "tar -xzf ~/dataset_v1_english.tar.gz -C ~/emotion-text"
```

### 13.5 方式 C：Git Bash / WSL 用 rsync，适合断点续传

如果你本机装了 Git Bash 或 WSL，可以用 `rsync`。它适合网络中断后继续传。

在 Git Bash 或 WSL 中执行：

```bash
cd /c/Users/Lenovo/Desktop/del

rsync -avh --progress \
  emotion_app scripts requirements.txt requirements-train.txt \
  ubuntu@100.64.12.34:~/emotion-text/

rsync -avh --progress \
  datasets/project-data/processed/dataset_v1 \
  ubuntu@100.64.12.34:~/emotion-text/datasets/project-data/processed/
```

如果端口不是 22：

```bash
rsync -avh --progress -e "ssh -p 2222" \
  emotion_app scripts requirements.txt requirements-train.txt \
  ubuntu@100.64.12.34:~/emotion-text/
```

### 13.6 上传后检查服务器目录

登录服务器：

```powershell
ssh ubuntu@100.64.12.34
```

在服务器执行：

```bash
cd ~/emotion-text
find . -maxdepth 4 -type f | sort | head -80
ls -lh
ls -lh datasets/project-data/processed/dataset_v1
```

必须能看到：

```text
emotion_app/
scripts/
requirements.txt
requirements-train.txt
datasets/project-data/processed/dataset_v1/train.csv
datasets/project-data/processed/dataset_v1/validation.csv
datasets/project-data/processed/dataset_v1/test.csv
datasets/project-data/processed/dataset_v1/manifest.json
```

## 14. 服务器首次配置完整命令

以下命令都在服务器上执行。

### 14.1 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl tmux htop unzip
```

检查 GPU：

```bash
nvidia-smi
```

应该能看到：

```text
NVIDIA GeForce RTX 3090
Driver Version: 595.71.05
CUDA Version: 13.2
```

### 14.2 创建 Python 虚拟环境

```bash
cd ~/emotion-text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 14.3 安装 PyTorch

优先安装 CUDA 12.8 wheel：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

如果这一步失败，去 PyTorch 官网复制当前 Linux + Pip + CUDA 的安装命令：

```text
https://pytorch.org/get-started/locally/
```

### 14.4 安装项目训练依赖

```bash
cd ~/emotion-text
source .venv/bin/activate
pip install -r requirements-train.txt
```

如果 `transformers` 不存在或版本太旧，补装：

```bash
pip install -U transformers accelerate sentencepiece protobuf pandas scikit-learn tqdm
```

### 14.5 验证 Python、PyTorch、GPU

```bash
python - <<'PY'
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("gpu count:", torch.cuda.device_count())
PY
```

必须看到：

```text
cuda available: True
gpu: NVIDIA GeForce RTX 3090
```

### 14.6 配置 Hugging Face 缓存

```bash
cd ~/emotion-text
mkdir -p cache/hf
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1
```

建议写入 `~/.bashrc`，以后登录自动生效：

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=$HOME/emotion-text/cache/hf
export HF_HUB_DISABLE_XET=1
EOF
```

重新加载：

```bash
source ~/.bashrc
```

### 14.7 预下载模型，避免训练时卡住

```bash
cd ~/emotion-text
source .venv/bin/activate
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1

python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
models = [
    "FacebookAI/xlm-roberta-large",
    "microsoft/mdeberta-v3-base",
    "hfl/chinese-roberta-wwm-ext-large",
]
for name in models:
    print("downloading", name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=7)
    print("ok", name)
print("all ok")
PY
```

如果 `hfl/chinese-roberta-wwm-ext-large` 下载较慢，可以先跳过，它只是中文诊断模型，不影响主线。

## 15. 数据校验和快速体检

在服务器执行：

```bash
cd ~/emotion-text
sha256sum datasets/project-data/processed/dataset_v1/train.csv
sha256sum datasets/project-data/processed/dataset_v1/validation.csv
sha256sum datasets/project-data/processed/dataset_v1/test.csv
```

必须匹配：

```text
9ab09538516b95da4ea58390feb9c42fffd6c560468f56c9664cd949bd0e2028  train.csv
5060da217def09fb6fcc277083ef776d18b9e58a1b4e99cf8ede1cb5f7b153bd  validation.csv
61339e070d23b051fb090717e5e8e7ea0c17c5bd22710cff1aeda1d82da00ba0  test.csv
```

如果输出带路径，只要哈希值一致即可。

再检查 CSV 列名和样本数：

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path("datasets/project-data/processed/dataset_v1")
for split in ["train", "validation", "test"]:
    df = pd.read_csv(root / f"{split}.csv")
    print(split, df.shape)
    print(df.columns.tolist())
    print(df["label"].value_counts().to_dict())
    print(df["language"].value_counts().to_dict())
PY
```

期望：

```text
train: 68102 行
validation: 8514 行
test: 8537 行
列名至少包含 text, label, language
```

## 16. 使用 tmux 防止 SSH 断开导致训练中断

服务器长时间训练一定建议用 `tmux`。

创建会话：

```bash
tmux new -s texttrain
```

进入后执行训练命令。断开 tmux 但保持训练继续：

```text
Ctrl+b 然后按 d
```

重新进入：

```bash
tmux attach -t texttrain
```

查看已有会话：

```bash
tmux ls
```

杀掉会话：

```bash
tmux kill-session -t texttrain
```

## 17. 第一轮必须先跑的 XLM-R large 基线

在服务器 `tmux` 里执行：

```bash
cd ~/emotion-text
source .venv/bin/activate
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1
mkdir -p outputs/text logs

RUN=xlmr_large_lr1e-5_len128_seed42
mkdir -p outputs/text/$RUN

python scripts/train_text_model.py \
  --data-dir datasets/project-data/processed/dataset_v1 \
  --model-name FacebookAI/xlm-roberta-large \
  --output-dir outputs/text/$RUN \
  --epochs 4 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --learning-rate 1e-5 \
  --max-length 128 \
  --seed 42 2>&1 | tee outputs/text/$RUN/train.log
```

另开一个 SSH 窗口监控 GPU：

```bash
watch -n 2 nvidia-smi
```

判断是否正常：

| 现象 | 说明 |
|---|---|
| GPU 显存占用上升 | 正常 |
| GPU 利用率周期性升高 | 正常 |
| 每个 epoch 后打印 validation 指标 | 正常 |
| 出现 CUDA out of memory | batch 太大，改为 `batch-size=1, gradient-accumulation=16` |
| GPU 利用率长期 0% | 可能没用上 CUDA 或卡在下载模型 |

## 18. 24 小时跑满服务器的推荐脚本

在服务器创建脚本：

```bash
cd ~/emotion-text
nano run_text_24h.sh
```

粘贴：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/emotion-text
source .venv/bin/activate
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1
mkdir -p outputs/text logs handoff

nvidia-smi > handoff/gpu-info.txt
pip freeze > handoff/pip-freeze.txt
cp datasets/project-data/processed/dataset_v1/manifest.json handoff/dataset-manifest.json

run_one() {
  local run="$1"
  local model="$2"
  local lr="$3"
  local len="$4"
  local seed="$5"
  local bs="$6"
  local ga="$7"

  if [ -f "outputs/text/$run/metrics.json" ]; then
    echo "skip existing $run"
    return
  fi

  mkdir -p "outputs/text/$run"
  echo "===== START $run ====="
  date
  nvidia-smi

  python scripts/train_text_model.py \
    --data-dir datasets/project-data/processed/dataset_v1 \
    --model-name "$model" \
    --output-dir "outputs/text/$run" \
    --epochs 4 \
    --batch-size "$bs" \
    --gradient-accumulation "$ga" \
    --learning-rate "$lr" \
    --max-length "$len" \
    --seed "$seed" 2>&1 | tee "outputs/text/$run/train.log"

  echo "===== END $run ====="
  date
}

run_one xlmr_large_lr1e-5_len128_seed42 FacebookAI/xlm-roberta-large 1e-5 128 42 2 8
run_one xlmr_large_lr2e-5_len128_seed42 FacebookAI/xlm-roberta-large 2e-5 128 42 2 8
run_one xlmr_large_lr1e-5_len192_seed42 FacebookAI/xlm-roberta-large 1e-5 192 42 2 8
run_one xlmr_large_lr1e-5_len128_seed123 FacebookAI/xlm-roberta-large 1e-5 128 123 2 8
run_one mdeberta_v3_base_lr1e-5_len128_seed42 microsoft/mdeberta-v3-base 1e-5 128 42 4 4
run_one mdeberta_v3_base_lr2e-5_len128_seed42 microsoft/mdeberta-v3-base 2e-5 128 42 4 4

python - <<'PY' | tee outputs/text/summary.md
import json
from pathlib import Path

rows = []
for p in sorted(Path("outputs/text").glob("*/metrics.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "run": p.parent.name,
        "model": m.get("model_name"),
        "val_macro_f1": m.get("best_validation_macro_f1"),
        "test_acc": m["test"]["accuracy"],
        "test_macro_f1": m["test"]["macro_f1"],
        "en_acc": m.get("test_en", {}).get("accuracy"),
        "en_macro_f1": m.get("test_en", {}).get("macro_f1"),
        "zh_acc": m.get("test_zh", {}).get("accuracy"),
        "zh_macro_f1": m.get("test_zh", {}).get("macro_f1"),
    })

rows.sort(key=lambda r: (r["val_macro_f1"] or 0), reverse=True)
print("| rank | run | model | val_macro_f1 | test_acc | test_macro_f1 | en_acc | en_macro_f1 | zh_acc | zh_macro_f1 |")
print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
for i, r in enumerate(rows, 1):
    def fmt(x):
        return "" if x is None else f"{x:.6f}"
    print(f"| {i} | {r['run']} | {r['model']} | {fmt(r['val_macro_f1'])} | {fmt(r['test_acc'])} | {fmt(r['test_macro_f1'])} | {fmt(r['en_acc'])} | {fmt(r['en_macro_f1'])} | {fmt(r['zh_acc'])} | {fmt(r['zh_macro_f1'])} |")
PY
```

保存 `nano`：

```text
Ctrl+O 回车
Ctrl+X
```

启动：

```bash
chmod +x run_text_24h.sh
nohup bash run_text_24h.sh > logs/run_text_24h.nohup.log 2>&1 &
tail -f logs/run_text_24h.nohup.log
```

如果你在 tmux 里运行，也可以不用 `nohup`：

```bash
bash run_text_24h.sh 2>&1 | tee logs/run_text_24h.tmux.log
```

## 19. 跑完一个实验后怎么快速看结果

```bash
cd ~/emotion-text
python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path("outputs/text").glob("*/metrics.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    print("\n", p.parent.name)
    print("model:", m.get("model_name"))
    print("best val macro_f1:", m.get("best_validation_macro_f1"))
    print("test accuracy:", m["test"]["accuracy"])
    print("test macro_f1:", m["test"]["macro_f1"])
    for key in ["test_en", "test_zh"]:
        if key in m:
            print(key, "acc", m[key]["accuracy"], "macro_f1", m[key]["macro_f1"])
PY
```

看逐类表现：

```bash
python - <<'PY'
import json
from pathlib import Path
best = sorted(Path("outputs/text").glob("*/metrics.json"))[-1]
m = json.loads(best.read_text(encoding="utf-8"))
print(best)
for label, item in m["test"]["classification_report"].items():
    if isinstance(item, dict) and "f1-score" in item:
        print(label, "precision", round(item["precision"], 4), "recall", round(item["recall"], 4), "f1", round(item["f1-score"], 4), "support", item["support"])
PY
```

## 20. 根据结果决定下一步

### 20.1 如果 `xlm-roberta-large` >= 75% Accuracy

说明大模型路线有效，下一步不要急着换模型。继续：

1. 跑更多 XLM-R large seed：`123, 456, 789`。
2. 保留 `lr=1e-5` 和 `lr=2e-5` 两条线。
3. 选 validation Macro-F1 前 3 的模型做 soft-voting ensemble。
4. 再考虑 class weight/focal loss。
5. 最后才考虑加入中文专用模型做中文分项补强。

建议追加命令：

```bash
for LR in 1e-5 2e-5; do
  for SEED in 123 456 789; do
    RUN=xlmr_large_lr${LR}_len128_seed${SEED}
    if [ -f outputs/text/$RUN/metrics.json ]; then continue; fi
    mkdir -p outputs/text/$RUN
    python scripts/train_text_model.py \
      --data-dir datasets/project-data/processed/dataset_v1 \
      --model-name FacebookAI/xlm-roberta-large \
      --output-dir outputs/text/$RUN \
      --epochs 4 \
      --batch-size 2 \
      --gradient-accumulation 8 \
      --learning-rate $LR \
      --max-length 128 \
      --seed $SEED 2>&1 | tee outputs/text/$RUN/train.log
  done
done
```

### 20.2 如果 `xlm-roberta-large` 在 72%-75%

说明有提升但离 80% 还远。优先补：

1. `max_length=160/192`。
2. `lr=8e-6/1e-5/2e-5`。
3. `mdeberta-v3-base` 对照。
4. 分析 `test_en` 和 `test_zh` 谁拖后腿。

如果中文拖后腿，跑中文诊断模型：

```bash
RUN=chinese_roberta_wwm_large_diag
mkdir -p outputs/text/$RUN
python scripts/train_text_model.py \
  --data-dir datasets/project-data/processed/dataset_v1 \
  --model-name hfl/chinese-roberta-wwm-ext-large \
  --output-dir outputs/text/$RUN \
  --epochs 4 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 2e-5 \
  --max-length 128 \
  --seed 42 2>&1 | tee outputs/text/$RUN/train.log
```

注意：中文模型用于诊断，不一定适合作为最终双语模型，因为英文数据会吃亏。

### 20.3 如果 `xlm-roberta-large` < 72%

不要继续盲目跑 XLM-R seed，立刻换 mDeBERTa：

```bash
for LR in 1e-5 2e-5; do
  RUN=mdeberta_v3_base_lr${LR}_len128_seed42
  mkdir -p outputs/text/$RUN
  python scripts/train_text_model.py \
    --data-dir datasets/project-data/processed/dataset_v1 \
    --model-name microsoft/mdeberta-v3-base \
    --output-dir outputs/text/$RUN \
    --epochs 4 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate $LR \
    --max-length 128 \
    --seed 42 2>&1 | tee outputs/text/$RUN/train.log
done
```

如果 mDeBERTa 也不到 72%，下一步重点应转向数据：

1. 抽查高置信错例。
2. 检查 `fear/disgust/anger` 标签质量。
3. 加 class weight 或 focal loss。
4. 只对 train 做弱类增强。
5. 必要时构建新的 `dataset_v2`，但不要和本轮 24 小时冲刺混在一起。

## 21. 结果回传到本机

### 21.1 服务器打包结果

在服务器执行：

```bash
cd ~/emotion-text
tar -czf text_runs_$(date +%Y%m%d_%H%M).tar.gz outputs/text handoff logs
ls -lh text_runs_*.tar.gz
```

如果模型文件太大，只想回传指标和日志：

```bash
cd ~/emotion-text
tar -czf text_metrics_logs_$(date +%Y%m%d_%H%M).tar.gz \
  outputs/text/*/metrics.json \
  outputs/text/*/confusion_matrix.csv \
  outputs/text/*/train.log \
  outputs/text/summary.md \
  handoff logs
```

### 21.2 本机下载结果

在本机 PowerShell 执行：

```powershell
cd C:\Users\Lenovo\Desktop\del
New-Item -ItemType Directory -Force server-results | Out-Null
scp ubuntu@100.64.12.34:~/emotion-text/text_runs_*.tar.gz server-results\
```

如果 SSH 端口不是 22：

```powershell
scp -P 2222 ubuntu@100.64.12.34:~/emotion-text/text_runs_*.tar.gz server-results\
```

下载后解压：

```powershell
cd C:\Users\Lenovo\Desktop\del\server-results
tar -xzf .\text_runs_*.tar.gz
```

### 21.3 只下载最佳模型

假设最佳 run 是：

```text
xlmr_large_lr1e-5_len128_seed42
```

本机执行：

```powershell
cd C:\Users\Lenovo\Desktop\del
New-Item -ItemType Directory -Force server-results\best-model | Out-Null
scp -r ubuntu@100.64.12.34:~/emotion-text/outputs/text/xlmr_large_lr1e-5_len128_seed42 server-results\best-model\
```

## 22. 常见问题处理

### 22.1 `CUDA out of memory`

把：

```text
--batch-size 2
--gradient-accumulation 8
```

改成：

```text
--batch-size 1
--gradient-accumulation 16
```

有效 batch size 不变，显存压力降低。

### 22.2 下载 Hugging Face 模型很慢

先设置：

```bash
export HF_HOME=$PWD/cache/hf
export HF_HUB_DISABLE_XET=1
```

如果仍然慢，可以先只跑主线模型：

```text
FacebookAI/xlm-roberta-large
microsoft/mdeberta-v3-base
```

中文诊断模型可以后面再下载。

### 22.3 SSH 断开

如果训练在 `tmux` 里，直接重新登录：

```bash
tmux attach -t texttrain
```

如果训练用 `nohup` 启动，查看：

```bash
tail -f ~/emotion-text/logs/run_text_24h.nohup.log
```

### 22.4 训练速度很慢但 GPU 没满

可以尝试：

```text
--batch-size 4
--gradient-accumulation 4
```

也可以确认没有跑在 CPU：

```bash
watch -n 2 nvidia-smi
```

### 22.5 `ModuleNotFoundError`

在服务器项目根目录执行：

```bash
cd ~/emotion-text
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -U transformers sentencepiece protobuf pandas scikit-learn tqdm
```

### 22.6 数据 hash 不一致

不要训练。重新上传：

```powershell
cd C:\Users\Lenovo\Desktop\del
scp -r datasets\project-data\processed\dataset_v1 ubuntu@100.64.12.34:~/emotion-text/datasets/project-data/processed/
```

## 23. 本轮服务器任务的最小成功标准

24 小时结束时，至少应该拿到：

| 产物 | 必须有吗 |
|---|---|
| `xlmr_large_lr1e-5_len128_seed42/metrics.json` | 必须 |
| `xlmr_large_lr2e-5_len128_seed42/metrics.json` | 强烈建议 |
| `mdeberta_v3_base_lr1e-5_len128_seed42/metrics.json` | 强烈建议 |
| `outputs/text/summary.md` | 必须 |
| `handoff/gpu-info.txt` | 必须 |
| `handoff/pip-freeze.txt` | 必须 |
| 每个 run 的 `train.log` | 必须 |

如果只跑完一个模型，也优先保证 `xlm-roberta-large lr=1e-5 seed=42` 完整跑完，因为它是判断后续方向的关键基线。
