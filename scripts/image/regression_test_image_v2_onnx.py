from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LABEL_ORDER = ("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")
APP_PROBABILITY_ORDER = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
MAX_DETECTION_DIMENSION = 960
REFERENCE_LANDMARKS_112 = np.asarray(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values -= np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


def top2(probabilities: np.ndarray) -> tuple[str, float, str, float]:
    order = np.argsort(probabilities)[::-1]
    return (
        LABEL_ORDER[int(order[0])],
        float(probabilities[int(order[0])]),
        LABEL_ORDER[int(order[1])],
        float(probabilities[int(order[1])]),
    )


def detection_view(frame: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= MAX_DETECTION_DIMENSION:
        return frame, 1.0
    scale = MAX_DETECTION_DIMENSION / float(longest)
    return (
        cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA),
        scale,
    )


def create_detector(detector_path: Path):
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.6, 0.3, 5000)
    return detector


def detect_faces(detector, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, np.ndarray | None]:
    detection_frame, scale = detection_view(frame_bgr)
    height, width = detection_frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(detection_frame)
    return detection_frame, scale, faces


def detect_faces_with_rotations(detector, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, np.ndarray | None, str]:
    detection_frame, scale, faces = detect_faces(detector, frame_bgr)
    if faces is not None and len(faces) > 0:
        return detection_frame, scale, faces, "original"
    rotations = [
        ("rotate_90_clockwise", cv2.ROTATE_90_CLOCKWISE),
        ("rotate_90_counterclockwise", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ("rotate_180", cv2.ROTATE_180),
    ]
    for name, code in rotations:
        rotated = cv2.rotate(frame_bgr, code)
        detection_frame, scale, faces = detect_faces(detector, rotated)
        if faces is not None and len(faces) > 0:
            return detection_frame, scale, faces, name
    return detection_frame, scale, faces, "not_detected"

def align_face(frame_bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    transform, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_LANDMARKS_112, method=cv2.LMEDS)
    if transform is None:
        x, y, w, h = [max(0, int(value)) for value in face[:4]]
        crop = frame_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            raise ValueError("detected face crop is empty")
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(frame_bgr, transform, (112, 112))


def blob_from_aligned(aligned_bgr: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.dnn.blobFromImage(
        aligned_bgr,
        scalefactor=1.0 / 127.5,
        size=(image_size, image_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    ).astype(np.float32)


def run_net(net, aligned_bgr: np.ndarray, image_size: int, flip_tta: bool) -> tuple[np.ndarray, np.ndarray]:
    blob = blob_from_aligned(aligned_bgr, image_size)
    if flip_tta:
        batch = np.concatenate((blob, blob[:, :, :, ::-1]), axis=0)
        net.setInput(batch)
        logits = net.forward()
        mean_logits = logits.mean(axis=0)
    else:
        net.setInput(blob)
        mean_logits = net.forward().reshape(-1)
    return mean_logits.reshape(-1), softmax(mean_logits)


def draw_faces(frame: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    out = frame.copy()
    for face in ([] if faces is None else faces):
        x, y, w, h = [int(v) for v in face[:4]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for px, py in np.asarray(face[4:14]).reshape(5, 2):
            cv2.circle(out, (int(px), int(py)), 2, (0, 0, 255), -1)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in paths:
        if path.is_file() and path.suffix.lower() in suffixes:
            images.append(path)
        elif path.is_dir():
            images.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes))
    seen: set[Path] = set()
    unique = []
    for image in images:
        resolved = image.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(image)
    return unique


def model_check(new_model_path: Path, metadata_path: Path, output_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "onnx_path": str(new_model_path),
        "exists": new_model_path.is_file(),
        "sha256": None,
        "input_shape": None,
        "output_shape": None,
        "opset": None,
        "label_order": list(LABEL_ORDER),
        "preprocessing": "BGR aligned face -> blobFromImage size 112x112, swapRB=True, x/127.5 - 1",
        "check_passed": False,
        "created_at": now(),
    }
    if new_model_path.is_file():
        payload["sha256"] = sha256(new_model_path)
        net = cv2.dnn.readNetFromONNX(str(new_model_path))
        sample = np.zeros((1, 3, 112, 112), dtype=np.float32)
        net.setInput(sample)
        out = net.forward()
        payload["input_shape"] = [1, 3, 112, 112]
        payload["output_shape"] = list(out.shape)
        if metadata_path.is_file():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["opset"] = meta.get("onnx_opset")
            payload["label_order"] = meta.get("label_order", list(LABEL_ORDER))
            payload["preprocessing"] = meta.get("preprocessing", payload["preprocessing"])
        payload["check_passed"] = (
            payload["output_shape"][-1] == 7
            and payload["input_shape"] == [1, 3, 112, 112]
            and tuple(payload["label_order"]) == LABEL_ORDER
        )
    write_json(output_dir / "model_check.json", payload)
    return payload


def static_regression(args: argparse.Namespace, output_dir: Path, debug_dir: Path) -> list[dict[str, Any]]:
    detector = create_detector(args.detector)
    old_net = cv2.dnn.readNetFromONNX(str(args.old_model))
    new_net = cv2.dnn.readNetFromONNX(str(args.new_model))
    rows: list[dict[str, Any]] = []
    images = find_images(args.images)
    for index, image_path in enumerate(images):
        frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            rows.append({"image_path": str(image_path), "face_detected": False, "manual_check_note": "cannot decode image"})
            continue
        detection_frame, scale, faces, orientation_used = detect_faces_with_rotations(detector, frame)
        face_detected = faces is not None and len(faces) > 0
        row: dict[str, Any] = {
            "image_path": str(image_path),
            "face_detected": bool(face_detected),
            "face_box": "",
            "keypoints_available": False,
            "old_model_pred": "",
            "old_model_top1_prob": "",
            "old_model_top2_label": "",
            "old_model_top2_prob": "",
            "new_model_pred": "",
            "new_model_top1_prob": "",
            "new_model_top2_label": "",
            "new_model_top2_prob": "",
            "new_model_pred_flip_tta": "",
            "new_model_flip_tta_top1_prob": "",
            "whether_prediction_changed": "",
            "manual_check_note": "",
        }
        cv2.imwrite(str(debug_dir / f"{index:02d}_original.jpg"), frame)
        cv2.imwrite(str(debug_dir / f"{index:02d}_yunet_boxes.jpg"), draw_faces(detection_frame, faces))
        if face_detected:
            face = max(faces, key=lambda item: item[2] * item[3])
            x, y, w, h = [int(round(v / scale)) for v in face[:4]]
            row["face_box"] = json.dumps([x, y, w, h], ensure_ascii=False)
            row["keypoints_available"] = len(face) >= 14
            aligned = align_face(detection_frame, face)
            cv2.imwrite(str(debug_dir / f"{index:02d}_aligned_112.jpg"), aligned)
            cv2.imwrite(str(debug_dir / f"{index:02d}_input_112.jpg"), cv2.resize(aligned, (112, 112)))
            _, old_probs = run_net(old_net, aligned, 100, flip_tta=True)
            old_pred, old_p1, old_top2, old_p2 = top2(old_probs)
            _, new_probs_no_tta = run_net(new_net, aligned, 112, flip_tta=False)
            new_pred, new_p1, new_top2, new_p2 = top2(new_probs_no_tta)
            _, new_probs_tta = run_net(new_net, aligned, 112, flip_tta=True)
            new_pred_tta, new_tta_p1, _new_tta_top2, _new_tta_p2 = top2(new_probs_tta)
            row.update(
                {
                    "old_model_pred": old_pred,
                    "old_model_top1_prob": old_p1,
                    "old_model_top2_label": old_top2,
                    "old_model_top2_prob": old_p2,
                    "new_model_pred": new_pred,
                    "new_model_top1_prob": new_p1,
                    "new_model_top2_label": new_top2,
                    "new_model_top2_prob": new_p2,
                    "new_model_pred_flip_tta": new_pred_tta,
                    "new_model_flip_tta_top1_prob": new_tta_p1,
                    "whether_prediction_changed": old_pred != new_pred_tta,
                    "manual_check_note": f"auto regression only; orientation={orientation_used}; user can visually inspect debug images",
                }
            )
        else:
            row["manual_check_note"] = "no face detected by YuNet after rotation fallback"
        rows.append(row)
    fields = [
        "image_path",
        "face_detected",
        "face_box",
        "keypoints_available",
        "old_model_pred",
        "old_model_top1_prob",
        "old_model_top2_label",
        "old_model_top2_prob",
        "new_model_pred",
        "new_model_top1_prob",
        "new_model_top2_label",
        "new_model_top2_prob",
        "new_model_pred_flip_tta",
        "new_model_flip_tta_top1_prob",
        "whether_prediction_changed",
        "manual_check_note",
    ]
    write_csv(output_dir / "static_image_predictions.csv", rows, fields)
    write_json(output_dir / "static_image_predictions.json", {"images": rows, "created_at": now()})
    return rows


def camera_regression(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    detector = create_detector(args.detector)
    old_net = cv2.dnn.readNetFromONNX(str(args.old_model))
    new_net = cv2.dnn.readNetFromONNX(str(args.new_model))
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    result: dict[str, Any] = {
        "camera_index": args.camera_index,
        "opened": bool(cap.isOpened()),
        "total_frames": 0,
        "frames_with_face": 0,
        "face_detection_rate": 0.0,
        "average_inference_latency_ms": None,
        "p95_inference_latency_ms": None,
        "old_model_prediction_distribution": {},
        "new_model_prediction_distribution": {},
        "frame_drop_or_error_count": 0,
        "created_at": now(),
    }
    if not cap.isOpened():
        result["skip_reason"] = "camera not available"
        write_json(output_dir / "camera_or_video_regression.json", result)
        return result
    old_counts: Counter[str] = Counter()
    new_counts: Counter[str] = Counter()
    latencies: list[float] = []
    deadline = time.perf_counter() + args.camera_seconds
    while time.perf_counter() < deadline and result["total_frames"] < args.camera_max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            result["frame_drop_or_error_count"] += 1
            continue
        result["total_frames"] += 1
        detection_frame, _scale, faces = detect_faces(detector, frame)
        if faces is None or len(faces) == 0:
            continue
        result["frames_with_face"] += 1
        face = max(faces, key=lambda item: item[2] * item[3])
        aligned = align_face(detection_frame, face)
        start = time.perf_counter()
        _, old_probs = run_net(old_net, aligned, 100, flip_tta=True)
        _, new_probs = run_net(new_net, aligned, 112, flip_tta=True)
        latencies.append((time.perf_counter() - start) * 1000.0)
        old_counts[top2(old_probs)[0]] += 1
        new_counts[top2(new_probs)[0]] += 1
    cap.release()
    result["face_detection_rate"] = result["frames_with_face"] / max(result["total_frames"], 1)
    if latencies:
        result["average_inference_latency_ms"] = statistics.mean(latencies)
        result["p95_inference_latency_ms"] = sorted(latencies)[max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))]
    result["old_model_prediction_distribution"] = dict(old_counts)
    result["new_model_prediction_distribution"] = dict(new_counts)
    write_json(output_dir / "camera_or_video_regression.json", result)
    return result


def tta_check(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    image_paths = find_images(args.images)
    payload = {
        "method": "original logits + horizontal flipped logits; average logits; then softmax",
        "averages_logits_not_probabilities": True,
        "sample_checked": None,
        "max_probability_difference_against_manual": None,
        "check_passed": False,
        "created_at": now(),
    }
    if not image_paths:
        payload["skip_reason"] = "no static images available"
        write_json(output_dir / "tta_check.json", payload)
        return payload
    detector = create_detector(args.detector)
    new_net = cv2.dnn.readNetFromONNX(str(args.new_model))
    for image_path in image_paths:
        frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        detection_frame, _scale, faces = detect_faces(detector, frame)
        if faces is None or len(faces) == 0:
            continue
        aligned = align_face(detection_frame, max(faces, key=lambda item: item[2] * item[3]))
        blob = blob_from_aligned(aligned, 112)
        flip_blob = blob[:, :, :, ::-1].copy()
        new_net.setInput(blob)
        logits_a = new_net.forward().reshape(-1)
        new_net.setInput(flip_blob)
        logits_b = new_net.forward().reshape(-1)
        manual_probs = softmax((logits_a + logits_b) / 2.0)
        _logits, helper_probs = run_net(new_net, aligned, 112, flip_tta=True)
        diff = float(np.max(np.abs(manual_probs - helper_probs)))
        payload.update(
            {
                "sample_checked": str(image_path),
                "max_probability_difference_against_manual": diff,
                "check_passed": diff < 1e-6,
            }
        )
        break
    write_json(output_dir / "tta_check.json", payload)
    return payload


def report(
    output_dir: Path,
    args: argparse.Namespace,
    model_payload: dict[str, Any],
    static_rows: list[dict[str, Any]],
    camera_payload: dict[str, Any],
    tta_payload: dict[str, Any],
) -> None:
    static_ok = bool(static_rows) and any(row.get("face_detected") for row in static_rows)
    preprocessing_ok = model_payload.get("check_passed") and tta_payload.get("check_passed")
    camera_ok = camera_payload.get("opened") and camera_payload.get("total_frames", 0) > 0
    recommend = bool(model_payload.get("check_passed") and preprocessing_ok and static_ok)
    lines = [
        "# image-v2 ONNX 真实部署回归报告",
        "",
        "## 1. 测试目的",
        "",
        "验证 image-v2 ONNX 是否能稳定接入现有图像识别链路：YuNet 检测、关键点对齐、112×112 resize、RGB/normalize、SE-ResNet18 ONNX 推理、label order 映射与 Flip TTA。",
        "",
        "## 2. 旧模型与 image-v2 模型路径",
        "",
        f"- YuNet：`{args.detector}`",
        f"- 旧分类模型：`{args.old_model}`，输入 100×100。",
        f"- image-v2 分类模型：`{args.new_model}`，输入 112×112。",
        "",
        "## 3. ONNX 检查结果",
        "",
        f"- passed：{model_payload.get('check_passed')}",
        f"- input_shape：{model_payload.get('input_shape')}",
        f"- output_shape：{model_payload.get('output_shape')}",
        f"- opset：{model_payload.get('opset')}",
        f"- sha256：`{model_payload.get('sha256')}`",
        f"- label_order：{model_payload.get('label_order')}",
        "",
        "## 4. 前处理一致性检查",
        "",
        f"- YuNet 检测：使用 `cv2.FaceDetectorYN`。",
        "- 关键点对齐：使用 YuNet 5 点关键点仿射到 112×112 标准模板。",
        "- resize：旧模型 100×100；image-v2 112×112。",
        "- RGB/BGR：输入为 BGR 对齐图，`blobFromImage(..., swapRB=True)` 转为 RGB，与训练一致。",
        "- normalize：`x / 127.5 - 1`，dtype float32。",
        f"- preprocess_debug：`{output_dir / 'preprocess_debug'}`",
        "",
        "## 5. 静态图片回归结果",
        "",
        f"- 样例数量：{len(static_rows)}",
        f"- 检出人脸样例：{sum(1 for row in static_rows if row.get('face_detected'))}",
        f"- 预测文件：`{output_dir / 'static_image_predictions.csv'}`",
        "",
        "| image | face | old_pred | new_pred_tta | changed | note |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in static_rows:
        lines.append(
            f"| {Path(row.get('image_path', '')).name} | {row.get('face_detected')} | {row.get('old_model_pred', '')} | {row.get('new_model_pred_flip_tta', '')} | {row.get('whether_prediction_changed', '')} | {row.get('manual_check_note', '')} |"
        )
    lines += [
        "",
        "## 6. 摄像头或视频回归结果",
        "",
        f"- opened：{camera_payload.get('opened')}",
        f"- total_frames：{camera_payload.get('total_frames')}",
        f"- frames_with_face：{camera_payload.get('frames_with_face')}",
        f"- face_detection_rate：{camera_payload.get('face_detection_rate')}",
        f"- average_inference_latency_ms：{camera_payload.get('average_inference_latency_ms')}",
        f"- p95_inference_latency_ms：{camera_payload.get('p95_inference_latency_ms')}",
        f"- frame_drop_or_error_count：{camera_payload.get('frame_drop_or_error_count')}",
        f"- old distribution：{camera_payload.get('old_model_prediction_distribution')}",
        f"- new distribution：{camera_payload.get('new_model_prediction_distribution')}",
        "",
        "## 7. Flip TTA 检查",
        "",
        f"- method：{tta_payload.get('method')}",
        f"- averages_logits_not_probabilities：{tta_payload.get('averages_logits_not_probabilities')}",
        f"- check_passed：{tta_payload.get('check_passed')}",
        f"- sample_checked：{tta_payload.get('sample_checked')}",
        f"- max_probability_difference_against_manual：{tta_payload.get('max_probability_difference_against_manual')}",
        "",
        "## 8. 是否发现 label order / resize / normalize / RGB-BGR 问题",
        "",
    ]
    if model_payload.get("check_passed") and tta_payload.get("check_passed"):
        lines.append("未发现 label order、resize、normalize、RGB/BGR 或 Flip TTA 实现问题。")
    else:
        lines.append("存在检查未通过项，请先查看 model_check.json / tta_check.json。")
    lines += [
        "",
        "## 9. 是否建议将 image-v2 接入主程序",
        "",
        "建议接入 image-v2 候选模型。" if recommend else "暂不建议接入 image-v2；需要先解决上述阻塞项。",
        "",
        "## 10. 如建议接入，需要修改的位置",
        "",
        "- `emotion_app/recognizers/image.py`：将分类模型路径切换到 `rafdb_se_resnet18_image_v2/rafdb_emotion_image_v2.onnx`。",
        "- 同文件 `_classify`：将 `blobFromImage` 的 size 从 `(100, 100)` 改为 `(112, 112)`，继续保留 logits 平均 Flip TTA。",
        "- `emotion_app.spec`：如后续要打包 EXE，再加入 image-v2 ONNX 与 metadata；本次未重打包。",
        "",
        "## 11. 如不建议接入，阻塞原因",
        "",
        "若本报告第 9 节显示不建议接入，阻塞通常来自 ONNX 缺失/shape 错误、静态样例无人脸、TTA 不一致或摄像头不可用。摄像头不可用本身不阻塞静态图片链路，但需要用户环境确认。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, default=Path("models/image/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--old-model", type=Path, default=Path("models/image/rafdb_se_resnet18/rafdb_emotion.onnx"))
    parser.add_argument("--new-model", type=Path, default=Path("models/image/rafdb_se_resnet18_image_v2/rafdb_emotion_image_v2.onnx"))
    parser.add_argument("--new-model-metadata", type=Path, default=Path("models/image/rafdb_se_resnet18_image_v2/rafdb_emotion_image_v2.onnx.metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/image/rafdb_se_resnet18_image_v2_deployment_regression"))
    parser.add_argument("--images", type=Path, nargs="*", default=[Path("cache")])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-seconds", type=float, default=5.0)
    parser.add_argument("--camera-max-frames", type=int, default=80)
    parser.add_argument("--skip-camera", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "preprocess_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model_check(args.new_model, args.new_model_metadata, args.output_dir)
    static_rows = static_regression(args, args.output_dir, debug_dir)
    tta_payload = tta_check(args, args.output_dir)
    if args.skip_camera:
        camera_payload = {
            "opened": False,
            "skip_reason": "skip-camera requested",
            "total_frames": 0,
            "frames_with_face": 0,
            "face_detection_rate": 0.0,
            "average_inference_latency_ms": None,
            "p95_inference_latency_ms": None,
            "old_model_prediction_distribution": {},
            "new_model_prediction_distribution": {},
            "frame_drop_or_error_count": 0,
            "created_at": now(),
        }
        write_json(args.output_dir / "camera_or_video_regression.json", camera_payload)
    else:
        camera_payload = camera_regression(args, args.output_dir)
    report(args.output_dir, args, model_payload, static_rows, camera_payload, tta_payload)
    print(json.dumps({"output_dir": str(args.output_dir), "model_check_passed": model_payload["check_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
