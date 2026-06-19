# -*- mode: python ; coding: utf-8 -*-
hiddenimports = [
    "transformers.models.bert.configuration_bert",
    "transformers.models.bert.modeling_bert",
    "transformers.models.bert.tokenization_bert",
    "transformers.models.xlm_roberta.configuration_xlm_roberta",
    "transformers.models.xlm_roberta.modeling_xlm_roberta",
    "transformers.models.xlm_roberta.tokenization_xlm_roberta",
    "tokenizers",
]
datas = []

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
        "tensorflow",
        "flask",
        "sqlalchemy",
        "sklearn",
        "scipy",
        "pytest",
        "matplotlib",
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
