from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
    RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RESOURCE_ROOT = PROJECT_ROOT


@dataclass(frozen=True)
class AppConfig:
    text_model_path: Path = Path(
        os.environ.get("EMOTION_TEXT_MODEL", RESOURCE_ROOT / "models" / "text")
    )
    max_text_length: int = int(os.environ.get("EMOTION_MAX_TEXT_LENGTH", "128"))
    batch_size: int = int(os.environ.get("EMOTION_BATCH_SIZE", "16"))
