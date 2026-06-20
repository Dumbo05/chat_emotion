from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from emotion_app.recognizers import ImageRecognizer, SpeechRecognizer, TextRecognizer
from emotion_app.ui.main_window import MainWindow


def main() -> int:
    text_recognizer = TextRecognizer()
    speech_recognizer = SpeechRecognizer()
    image_recognizer = ImageRecognizer()
    smoke_audio = os.environ.get("EMOTION_SPEECH_SMOKE_AUDIO")
    if smoke_audio:
        result = speech_recognizer.predict(smoke_audio)
        output = os.environ.get("EMOTION_SPEECH_SMOKE_OUTPUT")
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        return 0 if result.ok else 2
    smoke_image = os.environ.get("EMOTION_IMAGE_SMOKE")
    if smoke_image:
        result = image_recognizer.predict(smoke_image)
        output = os.environ.get("EMOTION_IMAGE_SMOKE_OUTPUT")
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        return 0 if result.ok else 2
    # PyTorch's native Windows runtime must be imported before QApplication is
    # constructed. Model weights are still loaded later in a QThread.
    if text_recognizer.available:
        try:
            text_recognizer.prepare_runtime()
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("中英双语情绪识别系统")
    app.setOrganizationName("EmotionRecognitionTeam")
    window = MainWindow(text_recognizer, speech_recognizer, image_recognizer)
    window.show()
    return app.exec_()



