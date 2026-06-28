# image-v4 24G empty-server runbook

This runbook assumes the GPU server is empty and you run commands from Windows PowerShell locally plus Bash on the server.

Replace these placeholders first:

- `<USER>`: server user name, for example `ubuntu`
- `<HOST>`: server IP or domain
- `<PORT>`: SSH port, usually `22`
- `<REMOTE_DIR>`: remote project directory, recommended `~/chat-emotion-image-v4`

## 0. What is being optimized

Final target/evaluation is still RAF-DB Basic because the current image module and report use RAF-DB as the benchmark. The change is that training is no longer RAF-only:

1. ImageNet-pretrained strong vision backbone from `timm`.
2. Optional large expression pretraining on AffectNet / FERPlus / ExpW / RAF-compatible ImageFolder data.
3. RAF-DB fine-tuning.
4. Multi-seed / multi-architecture soft-voting ensemble selected by RAF validation Macro-F1.

Current local reference:

| version | RAF-DB test Accuracy + TTA | RAF-DB test Macro-F1 + TTA |
| --- | ---: | ---: |
| deployed SE-ResNet18 | 78.16% | about 69% |
| image-v2 | 78.59% | 70.65% |
| image-v3 FER pretrain | 81.71% | 73.80% |

## 1. Local package and upload

Run on local Windows PowerShell from project root `C:\Users\Lenovo\Desktop\del`.

Create a small source archive. This excludes heavy existing outputs, build artifacts, virtual env, and old packaged app files, but keeps scripts, docs, requirements, and the RAF/FER data currently under `datasets/project-data`.

```powershell
cd C:\Users\Lenovo\Desktop\del
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pkg = "server-results\image_v4_source_$stamp.zip"
New-Item -ItemType Directory -Force server-results | Out-Null
Compress-Archive -Force -Path `
  app.py, emotion_app, scripts, docs, tests, requirements.txt, requirements-train.txt, pyproject.toml, README.md, TRAINING_GUIDE.md, datasets\project-data `
  -DestinationPath $pkg
Write-Host $pkg
```

Upload the archive:

```powershell
scp -P <PORT> $pkg <USER>@<HOST>:~/image_v4_source.zip
```

If your SSH port is 22, you can omit `-P <PORT>`:

```powershell
scp $pkg <USER>@<HOST>:~/image_v4_source.zip
```

## 2. Server base setup

Run on the server:

```bash
ssh -p <PORT> <USER>@<HOST>
```

Unpack project:

```bash
set -e
REMOTE_DIR=~/chat-emotion-image-v4
mkdir -p "$REMOTE_DIR"
unzip -o ~/image_v4_source.zip -d "$REMOTE_DIR"
cd "$REMOTE_DIR"
```

Check GPU:

```bash
nvidia-smi
python3 --version
```

Install system packages if needed:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip unzip git tmux htop
```

Create Python env:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install CUDA PyTorch first. For CUDA 12.1 servers:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If the server CUDA is 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install project training deps:

```bash
pip install -r requirements.txt -r requirements-train.txt timm
```

Verify:

```bash
python - <<'PY'
import torch, timm
print('cuda:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('timm:', timm.__version__)
PY
```

## 3. Dataset checks

RAF-DB should already be inside the uploaded package if local `datasets/project-data` was included:

```bash
ls datasets/project-data/processed/raf-db-basic/aligned | head
ls datasets/project-data/raw/raf-db-basic/extracted/EmoLabel/list_patition_label.txt
```

FER2013 local folder is named `fear2013` in this project. Check it:

```bash
find datasets/project-data/raw/fear2013 -maxdepth 2 -type d | sort | head -50
```

For true large expression pretraining, place additional datasets as ImageFolder roots. The script expects this layout:

```text
datasets/expression-pretrain/affectnet7/
  train/
    anger/...
    disgust/...
    fear/...
    happy/...
    sad/...
    surprise/...
    neutral/...
  val/              # or validation/ or test/
    anger/...
    ...
```

Accepted class folder names are mapped to RAF labels:

```text
angry, anger -> anger
disgust, disgusted -> disgust
fear, fearful -> fear
happy, happiness, joy -> joy
sad, sadness -> sadness
surprise, surprised -> surprise
neutral -> neutral
```

Ignored folders such as `contempt`, `unknown`, `non-face`, or compound labels are skipped.

If you later upload AffectNet/FERPlus/ExpW, put it under for example:

```bash
mkdir -p datasets/expression-pretrain
# after upload/unzip, expected path example:
# datasets/expression-pretrain/affectnet7/train/happy/*.jpg
# datasets/expression-pretrain/affectnet7/val/happy/*.jpg
```

## 4. Pre-download strong model weights

`timm` downloads ImageNet pretrained weights on first use, usually from Hugging Face. To warm the cache:

```bash
source .venv/bin/activate
python - <<'PY'
import timm
models = [
    'convnext_base.fb_in22k_ft_in1k',
    'tf_efficientnetv2_m.in21k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
]
for name in models:
    print('loading', name)
    model = timm.create_model(name, pretrained=True, num_classes=7)
    del model
print('done')
PY
```

If one model name fails because the installed `timm` version does not include it, list alternatives:

```bash
python - <<'PY'
import timm
for pat in ['convnext_base*', '*efficientnetv2*m*', 'vit_base_patch16_224*', 'swinv2_base*']:
    print('\n', pat)
    print('\n'.join(timm.list_models(pat, pretrained=True)[:30]))
PY
```

Then replace `--model-arch` with a listed model name.

## 5. Start long-running jobs

Use `tmux` so training survives disconnects:

```bash
tmux new -s imagev4
cd ~/chat-emotion-image-v4
source .venv/bin/activate
```

### 5.1 No extra big dataset yet: ImageNet + FER2013 + RAF

Run this immediately if the server only has the uploaded project data:

```bash
python scripts/image/train_rafdb_image_v4_strong.py \
  --model-arch convnext_base.fb_in22k_ft_in1k \
  --run-name convnext_base_224_seed42 \
  --seed 42 --image-size 224 --batch-size 48 --eval-batch-size 96 \
  --fer-epochs 8 --raf-epochs 45 --raf-learning-rate 8e-5
```

Second run:

```bash
python scripts/image/train_rafdb_image_v4_strong.py \
  --model-arch tf_efficientnetv2_m.in21k_ft_in1k \
  --run-name efficientnetv2_m_224_seed43 \
  --seed 43 --image-size 224 --batch-size 48 --eval-batch-size 96 \
  --fer-epochs 8 --raf-epochs 45 --raf-learning-rate 8e-5
```

Third run:

```bash
python scripts/image/train_rafdb_image_v4_strong.py \
  --model-arch vit_base_patch16_224.augreg_in21k_ft_in1k \
  --run-name vit_base_224_seed44 \
  --seed 44 --image-size 224 --batch-size 48 --eval-batch-size 96 \
  --fer-epochs 8 --raf-epochs 45 --raf-learning-rate 5e-5
```

If CUDA OOM happens, rerun with smaller batch:

```bash
# batch-size 48 -> 24, eval-batch-size 96 -> 48
```

### 5.2 With AffectNet/FERPlus/ExpW-style large expression dataset

Use `--expression-pretrain-root`; repeat it for multiple datasets.

```bash
python scripts/image/train_rafdb_image_v4_strong.py \
  --model-arch convnext_base.fb_in22k_ft_in1k \
  --run-name convnext_base_affectnet_seed42 \
  --seed 42 --image-size 224 --batch-size 48 --eval-batch-size 96 \
  --expression-pretrain-root datasets/expression-pretrain/affectnet7 \
  --fer-epochs 12 --expression-pretrain-learning-rate 3e-4 \
  --raf-epochs 45 --raf-learning-rate 8e-5
```

Multiple large expression datasets:

```bash
python scripts/image/train_rafdb_image_v4_strong.py \
  --model-arch convnext_base.fb_in22k_ft_in1k \
  --run-name convnext_base_bigexpr_seed42 \
  --seed 42 --image-size 224 --batch-size 48 --eval-batch-size 96 \
  --expression-pretrain-root datasets/expression-pretrain/affectnet7 \
  --expression-pretrain-root datasets/expression-pretrain/ferplus7 \
  --expression-pretrain-root datasets/expression-pretrain/expw7 \
  --fer-epochs 12 --expression-pretrain-learning-rate 3e-4 \
  --raf-epochs 45 --raf-learning-rate 8e-5
```

Detach tmux with `Ctrl-b` then `d`. Reattach:

```bash
tmux attach -t imagev4
```

## 6. Watch progress

Each run prints JSON rows. Key fields:

- `stage`: `expression_pretrain` then `raf_finetune`
- `val_macro_f1`: main selection metric
- `val_accuracy`: secondary metric

After a run finishes:

```bash
cat outputs/image/rafdb_image_v4_strong/runs/convnext_base_224_seed42/summary.json | python -m json.tool | head -120
```

Quick summary of all finished runs:

```bash
python - <<'PY'
from pathlib import Path
import json
for p in sorted(Path('outputs/image/rafdb_image_v4_strong/runs').glob('*/summary.json')):
    m=json.loads(p.read_text())
    print(p.parent.name, 'val_macro=', round(m['raf_val']['macro_f1'],4), 'val_acc=', round(m['raf_val']['accuracy'],4), 'test_acc=', round(m['raf_official_test']['accuracy'],4), 'test_macro=', round(m['raf_official_test']['macro_f1'],4))
PY
```

## 7. Ensemble

Run after at least 3 runs finish:

```bash
python scripts/image/ensemble_rafdb_image_v4.py \
  outputs/image/rafdb_image_v4_strong/runs/convnext_base_224_seed42 \
  outputs/image/rafdb_image_v4_strong/runs/efficientnetv2_m_224_seed43 \
  outputs/image/rafdb_image_v4_strong/runs/vit_base_224_seed44 \
  --max-members 5 \
  --output outputs/image/rafdb_image_v4_strong/ensemble
```

Read result:

```bash
cat outputs/image/rafdb_image_v4_strong/ensemble/ensemble_summary.csv
cat outputs/image/rafdb_image_v4_strong/ensemble/ensemble_results.json | python -m json.tool | head -160
```

## 8. Download results to local Windows

Run on local Windows PowerShell:

```powershell
cd C:\Users\Lenovo\Desktop\del
New-Item -ItemType Directory -Force server-results\image-v4 | Out-Null
scp -P <PORT> -r <USER>@<HOST>:~/chat-emotion-image-v4/outputs/image/rafdb_image_v4_strong server-results\image-v4\
```

If port is 22:

```powershell
scp -r <USER>@<HOST>:~/chat-emotion-image-v4/outputs/image/rafdb_image_v4_strong server-results\image-v4\
```

## 9. Practical target

90% on RAF-DB is ambitious. A realistic ladder is:

1. image-v3 baseline: 81.7% accuracy.
2. Strong single model: try to reach 84%-87%.
3. Multi-seed and architecture ensemble: try to add another 1%-3%.
4. If still below target, improve face crop/alignment and add better large expression data; more RAF-only training is unlikely to create a 90% jump.
