from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    text_model_path: Path = Path(
        os.environ.get("EMOTION_TEXT_MODEL", PROJECT_ROOT / "models" / "text")
    )
    max_text_length: int = int(os.environ.get("EMOTION_MAX_TEXT_LENGTH", "128"))
    batch_size: int = int(os.environ.get("EMOTION_BATCH_SIZE", "16"))

