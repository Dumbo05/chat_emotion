from __future__ import annotations

from pathlib import Path

from emotion_app.domain import RecognitionResult
from emotion_app.recognizers.base import FileRecognizerProtocol


class _UnavailableFileRecognizer(FileRecognizerProtocol):
    modality_zh = "文件"

    @property
    def available(self) -> bool:
        return False

    @property
    def status(self) -> str:
        return f"{self.modality_zh}模型尚未配置（一期预留接口）"

    def predict(self, path: str | Path) -> RecognitionResult:
        candidate = Path(path)
        if not candidate.is_file():
            return RecognitionResult.failure(f"请选择有效的{self.modality_zh}文件")
        return RecognitionResult.failure(self.status)


class SpeechRecognizer(_UnavailableFileRecognizer):
    """Extension point adapted from the upstream speech inference module."""

    modality_zh = "语音"


class ImageRecognizer(_UnavailableFileRecognizer):
    """Extension point adapted from the upstream image inference module."""

    modality_zh = "图像"

