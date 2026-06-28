# -*- mode: python ; coding: utf-8 -*-
hiddenimports = [
    "onnxruntime",
    "sklearn.pipeline",
    "sklearn.preprocessing._data",
    "sklearn.svm._classes",
    "sklearn.svm._base",
    "tokenizers",
    "emotion_app.recognizers.w2v_dann",
]

datas = [
    ("models/text/config.json", "models/text"),
    ("models/text/model.onnx", "models/text"),
    ("models/text/tokenizer.json", "models/text"),
    ("models/text/tokenizer_config.json", "models/text"),
    ("models/speech/wavlm_simsan_encoder.onnx", "models/speech"),
    ("models/speech/wavlm_simsan_head.joblib", "models/speech"),
    ("models/speech/wavlm_simsan_fixed_test_metrics.json", "models/speech"),
    ("models/speech/w2v_dann/w2v_dann_opt.onnx", "models/speech/w2v_dann"),
    ("models/speech/w2v_dann/w2v_dann_opt.onnx.data", "models/speech/w2v_dann"),
    ("models/speech/w2v_dann/server_eval_metrics.json", "models/speech/w2v_dann"),
    ("models/speech/speech_model.joblib", "models/speech"),
    ("models/speech/metrics.json", "models/speech"),
    ("models/speech/confusion_matrix.csv", "models/speech"),
    ("models/image/face_detection_yunet_2023mar.onnx", "models/image"),
    ("models/image/rafdb_v4_ensemble/efficientnetv2_m_224_seed42.onnx", "models/image/rafdb_v4_ensemble"),
    ("models/image/rafdb_v4_ensemble/convnext_large_224_seed42.onnx", "models/image/rafdb_v4_ensemble"),
    ("models/image/rafdb_v4_ensemble/maxvit_base_224_seed42.onnx", "models/image/rafdb_v4_ensemble"),
    ("models/image/rafdb_v4_ensemble/metadata.json", "models/image/rafdb_v4_ensemble"),
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow", "torch", "transformers", "safetensors",
        "flask", "sqlalchemy", "pytest", "matplotlib",
        "sklearn.tests", "scipy.tests",
    ],
    noarchive=False,
)
# PyQt5 ships an old VC++ 14.26 runtime. ONNX Runtime 1.27 requires the
# newer system VC++ runtime already collected at the application root.
# Keeping both makes Windows load the old DLL first when Qt initializes.
_vc_runtime_names = ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")
a.binaries = [
    item for item in a.binaries
    if not (
        item[0].lower().replace("/", "\\").startswith("pyqt5\\qt5\\bin\\")
        and item[0].lower().endswith(_vc_runtime_names)
    )
]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="EmotionRecognition",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
