from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEPENDENCIES = {
    # Native ML runtimes must be imported before PyQt5 on Windows.
    "PyTorch": "torch",
    "ONNX Runtime": "onnxruntime",
    "PyQt5": "PyQt5",
    "NumPy": "numpy",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "Transformers": "transformers",
    "SciPy": "scipy",
    "scikit-learn": "sklearn",
    "Joblib": "joblib",
    "miniaudio": "miniaudio",
    "OpenCV": "cv2",
}

REQUIRED_ASSETS = {
    "文本模型配置": Path("models/text/config.json"),
    "文本分词器": Path("models/text/tokenizer.json"),
    "图像人脸检测模型": Path("models/image/face_detection_yunet_2023mar.onnx"),
    "图像情感模型": Path("models/image/rafdb_se_resnet18/rafdb_emotion.onnx"),
    "语音 WavLM 编码器": Path("models/speech/wavlm_simsan_encoder.onnx"),
    "语音 SIMSAN 分类头": Path("models/speech/wavlm_simsan_head.joblib"),
}

EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}


def mark(ok: bool) -> str:
    return "[OK]" if ok else "[缺失]"


def check_dependencies() -> bool:
    print("\n1. Python 与依赖")
    version_ok = sys.version_info[:2] in {(3, 10), (3, 11), (3, 12)}
    print(f"{mark(version_ok)} Python {sys.version.split()[0]}（支持 3.10–3.12，推荐 3.11）")
    all_ok = version_ok
    for name, module in DEPENDENCIES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "已安装")
            print(f"[OK] {name}: {version}")
        except Exception as exc:
            all_ok = False
            print(f"[缺失] {name}: {exc}")
    return all_ok


def check_assets() -> bool:
    print("\n2. 运行时模型")
    all_ok = True
    for name, relative in REQUIRED_ASSETS.items():
        path = PROJECT_ROOT / relative
        ok = path.is_file() and path.stat().st_size > 0
        all_ok &= ok
        size = f"{path.stat().st_size / 1024 / 1024:.1f} MiB" if ok else "未找到"
        print(f"{mark(ok)} {name}: {relative.as_posix()} ({size})")

    text_dir = PROJECT_ROOT / "models" / "text"
    weight_candidates = (text_dir / "model.safetensors", text_dir / "pytorch_model.bin")
    weights_ok = any(path.is_file() and path.stat().st_size > 0 for path in weight_candidates)
    all_ok &= weights_ok
    print(f"{mark(weights_ok)} 文本模型权重: model.safetensors 或 pytorch_model.bin")

    config_path = text_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            labels = {str(value).lower() for value in config.get("id2label", {}).values()}
            labels_ok = labels == EMOTIONS
            all_ok &= labels_ok
            print(f"{mark(labels_ok)} 文本模型七分类标签完整")
        except Exception as exc:
            all_ok = False
            print(f"[缺失] config.json 无法解析: {exc}")
    return all_ok


def check_camera() -> bool:
    print("\n3. 默认摄像头")
    try:
        import cv2

        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        camera = cv2.VideoCapture(0, backend)
        opened = camera.isOpened()
        ok, frame = camera.read() if opened else (False, None)
        camera.release()
        passed = bool(opened and ok and frame is not None)
        print(f"{mark(passed)} 摄像头 0 可打开并读取画面")
        return passed
    except Exception as exc:
        print(f"[缺失] 摄像头检查失败: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 Chat Emotion 的运行环境与模型资产")
    parser.add_argument("--camera", action="store_true", help="同时检查默认摄像头 0")
    args = parser.parse_args()

    print(f"项目目录: {PROJECT_ROOT}")
    dependencies_ok = check_dependencies()
    assets_ok = check_assets()
    passed = dependencies_ok and assets_ok
    if args.camera:
        passed = check_camera() and passed

    print("\n检查结果:", "可以运行全部推理功能" if passed else "尚未满足全部运行条件")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
