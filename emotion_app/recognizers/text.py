from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from emotion_app.config import AppConfig
from emotion_app.domain import EMOTIONS, RecognitionResult, normalize_emotion
from emotion_app.recognizers.base import TextRecognizerProtocol


class ModelUnavailableError(RuntimeError):
    pass


class TextRecognizer(TextRecognizerProtocol):
    """Lazy, local-only Hugging Face text classifier.

    The recognizer intentionally has no heuristic fallback: missing or corrupt
    weights produce an explicit error instead of a fabricated prediction.
    """

    def __init__(self, model_path: str | Path | None = None, max_length: int | None = None):
        config = AppConfig()
        self.model_path = Path(model_path or config.text_model_path)
        self.max_length = max_length or config.max_text_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._auto_model_class: Any = None
        self._auto_tokenizer_class: Any = None
        self._index_to_emotion: dict[int, str] = {}
        self._load_error: str | None = None
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return self.model_path.is_dir() and (self.model_path / "config.json").is_file()

    @property
    def status(self) -> str:
        if self._model is not None:
            return f"已加载：{self.model_path.name}（{self._device}）"
        if self._load_error:
            return f"模型不可用：{self._load_error}"
        if self.available:
            return f"待加载：{self.model_path}"
        return f"未找到文本模型：{self.model_path}"

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise ModelUnavailableError(self.status)

        with self._lock:
            if self._model is not None:
                return
            try:
                self.prepare_runtime()
                tokenizer = self._auto_tokenizer_class.from_pretrained(
                    self.model_path, local_files_only=True
                )
                model = self._auto_model_class.from_pretrained(
                    self.model_path, local_files_only=True
                )
                raw_id2label = getattr(model.config, "id2label", {}) or {}
                index_to_emotion = {
                    int(index): normalize_emotion(label)
                    for index, label in raw_id2label.items()
                }
                if set(index_to_emotion.values()) != set(EMOTIONS):
                    raise ValueError(
                        "模型 config.json 的 id2label 必须完整包含七类标准标签"
                    )
                device = self._torch.device(
                    "cuda" if self._torch.cuda.is_available() else "cpu"
                )
                model.to(device)
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                self._device = device
                self._index_to_emotion = index_to_emotion
            except Exception as exc:
                self._load_error = str(exc)
                raise ModelUnavailableError(self._load_error) from exc

    def prepare_runtime(self) -> None:
        """Import the ML runtime before a Qt worker thread starts.

        On Windows, importing PyTorch for the first time inside a QThread can
        deadlock while native libraries initialize. The UI calls this once on
        the main thread; weight loading and inference remain asynchronous.
        """
        if self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._auto_model_class = AutoModelForSequenceClassification
            self._auto_tokenizer_class = AutoTokenizer
        except Exception as exc:
            self._load_error = str(exc)
            raise ModelUnavailableError(self._load_error) from exc

    def predict(self, text: str) -> RecognitionResult:
        content = str(text).strip()
        if not content:
            return RecognitionResult.failure("文本内容不能为空", self.model_path.name)
        try:
            self._load()
            encoded = self._tokenizer(
                content,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits[0]
                scores = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
            probabilities = {emotion: 0.0 for emotion in EMOTIONS}
            for index, score in enumerate(scores):
                probabilities[self._index_to_emotion[index]] = float(score)
            emotion = max(probabilities, key=probabilities.get)
            return RecognitionResult(
                emotion=emotion,
                confidence=probabilities[emotion],
                probabilities=probabilities,
                model_name=getattr(self._model.config, "name_or_path", self.model_path.name),
            )
        except Exception as exc:
            return RecognitionResult.failure(str(exc), self.model_path.name)
