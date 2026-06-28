# Chat Emotion: Local Multimodal Emotion Recognition

A PyQt5 desktop prototype for local emotion recognition from text, speech, and facial images. The project keeps a unified seven-class label space: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, and `neutral`.

This repository is intended for coursework, reproducible experiments, and engineering review. Source code, reports, plotting scripts, and small metadata are tracked. Large model weights, raw datasets, caches, server outputs, and packaged executables are intentionally excluded from Git.

## Current Modules

| Modality | Input | Main runtime method | Notes |
| --- | --- | --- | --- |
| Text | Chinese or English text / Excel rows | XLM-R style ONNX text classifier | Uses local `models/text` or `EMOTION_TEXT_MODEL` |
| Speech | WAV / MP3 | Wav2Vec2-DANN ONNX first, WavLM-SIMSAN fallback | Wav2Vec2-DANN reaches 89.38% on TESS but weak cross-dataset generalization |
| Image | Image file / camera frame | RAF-DB v4 ONNX ensemble | EfficientNetV2 + ConvNeXt-Large + MaxViT deployment path |

## Important Results

The latest integrated report is in [`docs/实验报告.md`](docs/实验报告.md). The report also includes Wav2Vec2-DANN reproduction and cross-dataset generalization appendices.

| Experiment | Dataset / protocol | Accuracy | Weighted or Macro F1 | Interpretation |
| --- | --- | ---: | ---: | --- |
| Text XLM-R | Seven-class text test set | 67.08% | 60.65% Macro-F1 | Baseline multilingual text model |
| Image v4 ensemble | RAF-DB official test | 89.41% | 83.17% Macro-F1 | Strong visual model, selected by validation Macro-F1 |
| Speech Wav2Vec2-DANN | TESS OAF -> YAF | 89.38% | 88.29% weighted F1 | Clean, same-corpus speaker-independent result |
| Speech Wav2Vec2-DANN | CREMA-D balanced sample | 21.33% | 17.71% weighted F1 | Cross-corpus generalization is weak |
| Speech Wav2Vec2-DANN | EmoDB mapped classes | 39.43% | 40.62% weighted F1 | Better than CREMA-D, still far below TESS |

## Repository Layout

```text
chat_emotion/
├── app.py                         # Desktop entry point
├── emotion_app/                   # Runtime app code, recognizers, UI, workers
│   ├── recognizers/
│   └── ui/
├── scripts/                       # Data preparation, training, evaluation, export, figures
│   ├── image/
│   ├── speech/
│   └── fusion/
├── tests/                         # Unit and smoke tests
├── docs/                          # Reports, figures, interface screenshots
├── datasets/                      # Small metadata and dataset instructions only
├── models/                        # README only; large weights are ignored
├── vendor/                        # Third-party licenses / minimal references
├── emotion_app.spec               # PyInstaller configuration
└── requirements*.txt              # Runtime, training, and dev dependencies
```

## Setup

Recommended: Windows 10/11 and Python 3.12 for the current local environment. Python 3.10+ should work for most scripts, but packaging should be tested on the target machine.

```powershell
git clone https://github.com/Dumbo05/chat_emotion.git
cd chat_emotion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Development and testing:

```powershell
pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

Training / export dependencies:

```powershell
pip install -r requirements-train.txt
```

## Model Files

Large model files are not committed. Prepare them under `models/` as described in [`models/README.md`](models/README.md). The desktop app will show a real error if a required model is missing; it does not use keyword heuristics or random fallback predictions.

Expected runtime locations include:

```text
models/text/
models/image/rafdb_v4_ensemble/
models/speech/w2v_dann/
models/speech/wavlm_simsan_encoder.onnx
models/speech/wavlm_simsan_head.joblib
```

## Run the Desktop App

```powershell
.\.venv\Scripts\python.exe app.py
```

If a local packaged build exists, run `dist/EmotionRecognition.exe`. Packaged executables are local release artifacts and are not pushed to GitHub.

## Useful Evaluation Commands

Wav2Vec2-DANN cross-dataset test:

```powershell
.\.venv\Scripts\python.exe scripts\speech\evaluate_w2v_dann_cross_dataset.py --dataset-path .\datasets\CREMA-D --dataset crema-d --max-per-class 50
.\.venv\Scripts\python.exe scripts\speech\evaluate_w2v_dann_cross_dataset.py --dataset-path .\datasets\EmoDB --dataset emodb
```

PyInstaller build:

```powershell
pip install pyinstaller
pyinstaller --noconfirm emotion_app.spec
```

## Data, Privacy, and Limitations

- Inference is local by default; the app does not upload user input.
- Raw datasets and pretrained checkpoints are governed by their original licenses and are not redistributed here.
- Emotion recognition is affected by culture, language, context, microphone/camera quality, and annotation ambiguity.
- Reported dataset metrics should not be interpreted as medical, psychological, hiring, legal, or real-world diagnostic evidence.

## License

Project code is released under the [MIT License](LICENSE). Third-party notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
