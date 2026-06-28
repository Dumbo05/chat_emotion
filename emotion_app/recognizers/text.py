from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from emotion_app.config import AppConfig
from emotion_app.domain import EMOTIONS, RecognitionResult, normalize_emotion
from emotion_app.recognizers.base import TextRecognizerProtocol


class ModelUnavailableError(RuntimeError):
    pass


class TextRecognizer(TextRecognizerProtocol):
    """Local-only ONNX text classifier.

    The desktop app uses the exported ONNX model instead of requiring PyTorch
    or Transformers at runtime. Missing or corrupt weights produce an explicit
    error instead of a fabricated prediction.
    """

    def __init__(self, model_path: str | Path | None = None, max_length: int | None = None):
        config = AppConfig()
        self.model_path = Path(model_path or config.text_model_path)
        self.max_length = max_length or config.max_text_length
        self._tokenizer: Any = None
        self._session: Any = None
        self._index_to_emotion: dict[int, str] = {}
        self._load_error: str | None = None
        self._ort: Any = None
        self._tokenizer_class: Any = None
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return (
            self.model_path.is_dir()
            and (self.model_path / "config.json").is_file()
            and (self.model_path / "tokenizer.json").is_file()
            and (self.model_path / "model.onnx").is_file()
        )

    @property
    def status(self) -> str:
        if self._session is not None:
            return f"已加载：{self.model_path.name}（ONNX Runtime CPU）"
        if self._load_error:
            return f"模型不可用：{self._load_error}"
        if self.available:
            return f"待加载：{self.model_path}"
        return f"未找到文本模型：{self.model_path}"

    def prepare_runtime(self) -> None:
        if self._ort is not None and self._tokenizer_class is not None:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._ort = ort
            self._tokenizer_class = Tokenizer
        except Exception as exc:
            self._load_error = str(exc)
            raise ModelUnavailableError(self._load_error) from exc

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self.available:
            raise ModelUnavailableError(self.status)

        with self._lock:
            if self._session is not None:
                return
            try:
                self.prepare_runtime()
                with (self.model_path / "config.json").open("r", encoding="utf-8") as handle:
                    config = json.load(handle)
                raw_id2label = config.get("id2label", {}) or {}
                index_to_emotion = {
                    int(index): normalize_emotion(label)
                    for index, label in raw_id2label.items()
                }
                if set(index_to_emotion.values()) != set(EMOTIONS):
                    raise ValueError("模型 config.json 的 id2label 必须完整包含七类标准标签")

                tokenizer = self._tokenizer_class.from_file(str(self.model_path / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=self.max_length)
                session = self._ort.InferenceSession(
                    str(self.model_path / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                self._tokenizer = tokenizer
                self._session = session
                self._index_to_emotion = index_to_emotion
            except Exception as exc:
                self._load_error = str(exc)
                raise ModelUnavailableError(self._load_error) from exc

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        values = logits.astype(np.float64)
        values -= np.max(values)
        exp = np.exp(values)
        return exp / np.sum(exp)

    def predict(self, text: str) -> RecognitionResult:
        content = str(text).strip()
        if not content:
            return RecognitionResult.failure("文本内容不能为空", self.model_path.name)
        try:
            self._load()
            encoded = self._tokenizer.encode(content)
            input_ids = np.asarray([encoded.ids], dtype=np.int64)
            attention_mask = np.asarray([encoded.attention_mask], dtype=np.int64)
            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            scores = self._softmax(np.asarray(outputs[0][0]))
            probabilities = {emotion: 0.0 for emotion in EMOTIONS}
            for index, score in enumerate(scores):
                probabilities[self._index_to_emotion[index]] = float(score)
            emotion = max(probabilities, key=probabilities.get)
            return RecognitionResult(
                emotion=emotion,
                confidence=probabilities[emotion],
                probabilities=probabilities,
                model_name=self.model_path.name,
            )
        except Exception as exc:
            return RecognitionResult.failure(str(exc), self.model_path.name)
