from __future__ import annotations

from pathlib import Path

import numpy as np

from emotion_app.audio_features import TARGET_SAMPLE_RATE, _read_audio
from emotion_app.domain import EMOTIONS, RecognitionResult


MODEL_NAME = "Wav2Vec2-DANN Optimized"
CLASS_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad")
CLASS_TO_EMOTION = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "neutral": "neutral",
    "pleasant_surprise": "surprise",
    "sad": "sadness",
}


class Wav2VecDANNPredictor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / "w2v_dann_opt.onnx"
        self._session = None

    @property
    def available(self) -> bool:
        return self.onnx_path.is_file()

    def _load(self):
        if self._session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64)
        values -= values.max(axis=-1, keepdims=True)
        exp = np.exp(values)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def _waveform(path: Path) -> np.ndarray:
        signal = _read_audio(path, target_rate=TARGET_SAMPLE_RATE).astype(np.float32)
        max_len = TARGET_SAMPLE_RATE * 3
        if signal.shape[0] < max_len:
            signal = np.pad(signal, (0, max_len - signal.shape[0]))
        else:
            signal = signal[:max_len]
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        return signal[None, :].astype(np.float32)

    def predict(self, path: str | Path) -> RecognitionResult:
        session = self._load()
        logits = session.run(None, {"input_values": self._waveform(Path(path))})[0]
        values = self._softmax(logits)[0]
        probabilities = {emotion: 0.0 for emotion in EMOTIONS}
        for label, probability in zip(CLASS_NAMES, values):
            probabilities[CLASS_TO_EMOTION[label]] += float(probability)
        emotion = max(probabilities, key=probabilities.get)
        return RecognitionResult(emotion, probabilities[emotion], probabilities, MODEL_NAME)
