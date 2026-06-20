# -*- mode: python ; coding: utf-8 -*-
hiddenimports = [
    "sklearn.pipeline",
    "sklearn.preprocessing._data",
    "sklearn.svm._classes",
    "sklearn.svm._base",
]

datas = [
    ("models/speech/speech_model.joblib", "models/speech"),
    ("models/speech/metrics.json", "models/speech"),
    ("models/speech/confusion_matrix.csv", "models/speech"),
    ("models/image/face_detection_yunet_2023mar.onnx", "models/image"),
    ("models/image/facial_expression_recognition_mobilefacenet_2022july.onnx", "models/image"),
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
        "tensorflow", "torch", "transformers", "tokenizers",
        "flask", "sqlalchemy", "pytest", "matplotlib",
        "sklearn.tests", "scipy.tests",
    ],
    noarchive=False,
)
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



