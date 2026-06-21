from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np

from emotion_app.audio_features import _read_audio, extract_audio_features, select_speaker_reduced_features
from emotion_app.config import RESOURCE_ROOT
from emotion_app.domain import EMOTIONS, RecognitionResult, normalize_emotion
from emotion_app.recognizers.base import FileRecognizerProtocol

MODEL_NAME = "WavLM-SIMSAN"

class SpeechRecognizer(FileRecognizerProtocol):
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or RESOURCE_ROOT / "models" / "speech")
        self._encoder = self._bundle = self._legacy_model = None
        self._dll_directory = None
        self._model = None  # legacy test/API compatibility

    @property
    def _encoder_path(self): return self.model_path / "wavlm_simsan_encoder.onnx"
    @property
    def _head_path(self): return self.model_path / "wavlm_simsan_head.joblib"

    @property
    def available(self) -> bool:
        return (self._encoder_path.is_file() and self._head_path.is_file()) or (self.model_path / "speech_model.joblib").is_file()

    @property
    def metrics(self) -> dict:
        path = self.model_path / "wavlm_simsan_fixed_test_metrics.json"
        if not path.is_file(): path = self.model_path / "metrics.json"
        try: return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError): return {}

    @property
    def status(self) -> str:
        if not self.available: return f"未找到语音模型：{self.model_path}（请先运行训练脚本）"
        if self._encoder_path.is_file() and self._head_path.is_file(): return "WavLM-SIMSAN 跨说话人模型已就绪（支持 WAV、MP3）"
        return "多数据集 MFCC-SVM 模型已就绪（支持 WAV、MP3）"

    def _load_best(self):
        if self._encoder is None:
            if getattr(sys, "frozen", False):
                capi = Path(sys._MEIPASS) / "onnxruntime" / "capi"
                self._dll_directory = os.add_dll_directory(str(capi))
            import onnxruntime as ort
            options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
            self._encoder = ort.InferenceSession(str(self._encoder_path), sess_options=options, providers=["CPUExecutionProvider"])
            self._bundle = joblib.load(self._head_path)
        return self._encoder, self._bundle

    @staticmethod
    def _waveform(path: Path) -> np.ndarray:
        signal = _read_audio(path)[:64_000].astype(np.float32)
        signal = (signal - signal.mean()) / (signal.std() + 1e-7)
        return signal[None, :]

    def _predict_best(self, path: Path) -> RecognitionResult:
        encoder, bundle = self._load_best()
        features = encoder.run(None, {"input_values": self._waveform(path)})[0]
        model = bundle["model"]
        scores = np.asarray(model.decision_function(features)[0], dtype=np.float64); scores -= scores.max()
        raw = np.exp(scores); raw /= raw.sum()
        labels = bundle.get("labels", EMOTIONS)
        probabilities = {emotion: 0.0 for emotion in EMOTIONS}
        for class_id, probability in zip(model.classes_, raw):
            label = labels[int(class_id)] if isinstance(class_id, (int, np.integer)) else class_id
            probabilities[normalize_emotion(str(label))] = float(probability)
        emotion = max(probabilities, key=probabilities.get)
        return RecognitionResult(emotion, probabilities[emotion], probabilities, MODEL_NAME)

    def _predict_legacy(self, path: Path) -> RecognitionResult:
        if self._model is None: self._model = joblib.load(self.model_path / "speech_model.joblib")
        self._legacy_model = self._model
        raw = self._legacy_model.predict_proba(select_speaker_reduced_features(extract_audio_features(path))[None, :])[0]
        probabilities = {emotion: 0.0 for emotion in EMOTIONS}
        for label, probability in zip(self._legacy_model.classes_, raw): probabilities[normalize_emotion(str(label))] = float(probability)
        emotion = max(probabilities, key=probabilities.get)
        return RecognitionResult(emotion, probabilities[emotion], probabilities, "多数据集 MFCC-SVM")

    def predict(self, path: str | Path) -> RecognitionResult:
        candidate = Path(path)
        if not candidate.is_file(): return RecognitionResult.failure("请选择有效的 WAV 或 MP3 音频文件", MODEL_NAME)
        if candidate.suffix.lower() not in {".wav", ".mp3"}: return RecognitionResult.failure("当前语音模型仅支持 WAV 或 MP3 音频", MODEL_NAME)
        if not self.available: return RecognitionResult.failure(self.status, MODEL_NAME)
        try:
            return self._predict_best(candidate) if self._encoder_path.is_file() and self._head_path.is_file() else self._predict_legacy(candidate)
        except Exception as exc:
            return RecognitionResult.failure(f"语音识别失败：{exc}", MODEL_NAME)




