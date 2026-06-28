# Model Directory

Large runtime weights and training checkpoints are intentionally not tracked by Git. Put local model artifacts under this directory when you run or package the app.

Expected runtime layout:

```text
models/
├── text/
│   ├── model.onnx
│   ├── config.json
│   ├── tokenizer.json / tokenizer files
│   └── optional Hugging Face metadata
├── image/
│   └── rafdb_v4_ensemble/
│       ├── efficientnetv2_m_224_seed42.onnx
│       ├── convnext_large_224_seed42.onnx
│       ├── maxvit_base_224_seed42.onnx
│       └── metadata.json
└── speech/
    ├── w2v_dann/
    │   ├── w2v_dann_opt.onnx
    │   ├── w2v_dann_opt.onnx.data
    │   └── server_eval_metrics.json
    ├── wavlm_simsan_encoder.onnx
    ├── wavlm_simsan_head.joblib
    └── speech_model.joblib              # optional legacy fallback
```

The text model can also be redirected with `EMOTION_TEXT_MODEL`.

The current speech recognizer prefers Wav2Vec2-DANN if `models/speech/w2v_dann/w2v_dann_opt.onnx` exists, then falls back to WavLM-SIMSAN, then to the legacy speech model.

Do not commit raw datasets, pretrained checkpoints, `.pth`, `.safetensors`, `.onnx`, `.onnx.data`, `.joblib`, or packaged `.exe` files unless you intentionally move them to an external release or model registry with the correct license.
