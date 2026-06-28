from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

from emotion_app.config import RESOURCE_ROOT
from emotion_app.domain import EMOTIONS, RecognitionResult
from emotion_app.recognizers.base import FileRecognizerProtocol

MODEL_NAME = "RAF-DB v4 EfficientNetV2 + ConvNeXt-Large + MaxViT ensemble"
MODEL_LABELS = ("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")
ENSEMBLE_MEMBERS = (
    "efficientnetv2_m_224_seed42.onnx",
    "convnext_large_224_seed42.onnx",
    "maxvit_base_224_seed42.onnx",
)
INPUT_SIZE = 224
MAX_DETECTION_DIMENSION = 960
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class FacePrediction:
    box: tuple[int, int, int, int]
    result: RecognitionResult


class ImageRecognizer(FileRecognizerProtocol):
    """Seven-class facial-expression recognizer for images and BGR camera frames."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or RESOURCE_ROOT / "models" / "image")
        self.detector_path = self.model_path / "face_detection_yunet_2023mar.onnx"
        self.expression_dir = self.model_path / "rafdb_v4_ensemble"
        self.expression_paths = [self.expression_dir / name for name in ENSEMBLE_MEMBERS]
        self._detector = None
        self._expression_sessions: list[ort.InferenceSession] = []
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return self.detector_path.is_file() and all(path.is_file() for path in self.expression_paths)

    @property
    def status(self) -> str:
        if not self.available:
            return f"Image model files are missing under: {self.model_path}"
        return f"{MODEL_NAME} is ready for image files and camera frames"

    def _load(self) -> None:
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.detector_path), "", (320, 320), 0.6, 0.3, 5000
            )
        if not self._expression_sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._expression_sessions = [
                ort.InferenceSession(
                    str(path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                for path in self.expression_paths
            ]

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        values -= np.max(values)
        exp = np.exp(values)
        return exp / exp.sum()

    @staticmethod
    def _align_face(frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
        reference = np.asarray(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        transform, _ = cv2.estimateAffinePartial2D(landmarks, reference, method=cv2.LMEDS)
        if transform is None:
            x, y, w, h = [max(0, int(value)) for value in face[:4]]
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                raise ValueError("Detected face crop is empty")
            return cv2.resize(crop, (112, 112))
        return cv2.warpAffine(frame, transform, (112, 112))

    @staticmethod
    def _preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(aligned_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

    def _classify(self, aligned_bgr: np.ndarray) -> RecognitionResult:
        original = self._preprocess(aligned_bgr)
        flipped = self._preprocess(cv2.flip(aligned_bgr, 1))
        batch = np.stack([original, flipped], axis=0)

        probabilities_by_model = []
        for session in self._expression_sessions:
            input_name = session.get_inputs()[0].name
            logits = session.run(None, {input_name: batch})[0].mean(axis=0)
            probabilities_by_model.append(self._softmax(logits))
        values = np.mean(probabilities_by_model, axis=0)

        probabilities = {emotion: 0.0 for emotion in EMOTIONS}
        for label, probability in zip(MODEL_LABELS, values):
            probabilities[label] = float(probability)
        emotion = max(probabilities, key=probabilities.get)
        return RecognitionResult(emotion, probabilities[emotion], probabilities, MODEL_NAME)

    @staticmethod
    def _detection_view(frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest <= MAX_DETECTION_DIMENSION:
            return frame, 1.0
        scale = MAX_DETECTION_DIMENSION / float(longest)
        resized = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _predict_orientation(self, frame: np.ndarray) -> list[FacePrediction]:
        detection_frame, scale = self._detection_view(frame)
        height, width = detection_frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(detection_frame)
        predictions: list[FacePrediction] = []
        for face in ([] if faces is None else faces):
            result = self._classify(self._align_face(detection_frame, face))
            x, y, w, h = [int(round(value / scale)) for value in face[:4]]
            predictions.append(FacePrediction((x, y, w, h), result))
        return predictions

    @staticmethod
    def _box_to_original(
        box: tuple[int, int, int, int], rotation: int, original_shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        x, y, width, height = box
        original_height, original_width = original_shape[:2]
        if rotation == cv2.ROTATE_90_CLOCKWISE:
            mapped = (y, original_height - x - width, height, width)
        elif rotation == cv2.ROTATE_180:
            mapped = (
                original_width - x - width,
                original_height - y - height,
                width,
                height,
            )
        elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            mapped = (original_width - y - height, x, height, width)
        else:
            mapped = box
        mx, my, mw, mh = mapped
        return (
            max(0, min(original_width - 1, mx)),
            max(0, min(original_height - 1, my)),
            max(1, min(original_width, mw)),
            max(1, min(original_height, mh)),
        )

    def predict_frame(
        self, frame: np.ndarray, *, try_rotations: bool = False
    ) -> list[FacePrediction]:
        if not self.available:
            raise RuntimeError(self.status)
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("Camera or image frame is invalid")
        if frame.shape[0] < 20 or frame.shape[1] < 20:
            raise ValueError("Image frame is too small")
        with self._lock:
            self._load()
            predictions = self._predict_orientation(frame)
            if predictions or not try_rotations:
                return predictions
            for rotation in (
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180,
            ):
                rotated = cv2.rotate(frame, rotation)
                rotated_predictions = self._predict_orientation(rotated)
                if rotated_predictions:
                    return [
                        FacePrediction(
                            self._box_to_original(item.box, rotation, frame.shape),
                            item.result,
                        )
                        for item in rotated_predictions
                    ]
            return []

    def predict(self, path: str | Path) -> RecognitionResult:
        candidate = Path(path)
        if not candidate.is_file():
            return RecognitionResult.failure("Please choose a valid image file", MODEL_NAME)
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return RecognitionResult.failure("Only PNG, JPG, JPEG, BMP and WebP images are supported", MODEL_NAME)
        if not self.available:
            return RecognitionResult.failure(self.status, MODEL_NAME)
        try:
            frame = cv2.imdecode(np.fromfile(candidate, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return RecognitionResult.failure("Cannot read the image file", MODEL_NAME)
            faces = self.predict_frame(frame, try_rotations=True)
            if not faces:
                return RecognitionResult.failure(
                    "No face was detected in the image. Use a clear, frontal, unobstructed face photo.",
                    MODEL_NAME,
                )
            return max(faces, key=lambda item: item.box[2] * item.box[3]).result
        except Exception as exc:
            return RecognitionResult.failure(f"Image recognition failed: {exc}", MODEL_NAME)
