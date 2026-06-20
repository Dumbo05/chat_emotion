from __future__ import annotations

import json
from pathlib import Path
import joblib

from emotion_app.audio_features import extract_audio_features
from emotion_app.config import RESOURCE_ROOT
from emotion_app.domain import EMOTIONS, RecognitionResult, normalize_emotion
from emotion_app.recognizers.base import FileRecognizerProtocol


class SpeechRecognizer(FileRecognizerProtocol):
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or RESOURCE_ROOT / "models" / "speech")
        self._model = None

    @property
    def available(self) -> bool:
        return (self.model_path / "speech_model.joblib").is_file()

    @property
    def metrics(self) -> dict:
        path = self.model_path / "metrics.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @property
    def status(self) -> str:
        if not self.available:
            return f"未找到语音模型：{self.model_path}（请先运行训练脚本）"
        return "TESS MFCC-SVM 模型已就绪（支持 WAV、MP3）"

    def _load(self):
        if self._model is None:
            self._model = joblib.load(self.model_path / "speech_model.joblib")
        return self._model

    def predict(self, path: str | Path) -> RecognitionResult:
        candidate = Path(path)
        if not candidate.is_file():
            return RecognitionResult.failure("请选择有效的 WAV 或 MP3 音频文件", "TESS MFCC-SVM")
        if candidate.suffix.lower() not in {".wav", ".mp3"}:
            return RecognitionResult.failure("当前语音模型仅支持 WAV 或 MP3 音频", "TESS MFCC-SVM")
        if not self.available:
            return RecognitionResult.failure(self.status, "TESS MFCC-SVM")
        try:
            model = self._load()
            raw = model.predict_proba(extract_audio_features(candidate)[None, :])[0]
            probabilities = {emotion: 0.0 for emotion in EMOTIONS}
            for label, probability in zip(model.classes_, raw):
                probabilities[normalize_emotion(str(label))] = float(probability)
            emotion = max(probabilities, key=probabilities.get)
            return RecognitionResult(emotion, probabilities[emotion], probabilities, "TESS MFCC-SVM")
        except Exception as exc:
            return RecognitionResult.failure(f"语音识别失败：{exc}", "TESS MFCC-SVM")




