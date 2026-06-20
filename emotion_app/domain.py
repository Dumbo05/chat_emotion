from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
EMOTION_LABELS_ZH = {
    "anger": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "joy": "喜悦",
    "sadness": "悲伤",
    "surprise": "惊讶",
    "neutral": "中性",
}
LABEL_ALIASES = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}


def normalize_emotion(label: str) -> str:
    normalized = LABEL_ALIASES.get(str(label).strip().lower())
    if normalized is None:
        raise ValueError(f"不支持的情绪标签：{label}")
    return normalized


@dataclass(frozen=True)
class RecognitionResult:
    emotion: str | None
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)
    model_name: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.emotion in EMOTIONS

    @classmethod
    def failure(cls, error: str, model_name: str = "") -> "RecognitionResult":
        return cls(None, 0.0, {}, model_name, error)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["emotion_zh"] = EMOTION_LABELS_ZH.get(self.emotion or "", "")
        payload["ok"] = self.ok
        return payload

