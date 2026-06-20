from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from emotion_app.domain import RecognitionResult


class Recognizer(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether a real model is configured and ready to be loaded."""

    @property
    @abstractmethod
    def status(self) -> str:
        """Human-readable model status."""


class TextRecognizerProtocol(Recognizer):
    @abstractmethod
    def predict(self, text: str) -> RecognitionResult:
        raise NotImplementedError


class FileRecognizerProtocol(Recognizer):
    @abstractmethod
    def predict(self, path: str | Path) -> RecognitionResult:
        raise NotImplementedError

