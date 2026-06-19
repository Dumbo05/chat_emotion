from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from emotion_app.recognizers import ImageRecognizer, SpeechRecognizer, TextRecognizer
from emotion_app.ui.main_window import MainWindow


def main() -> int:
    text_recognizer = TextRecognizer()
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
    window = MainWindow(text_recognizer, SpeechRecognizer(), ImageRecognizer())
    window.show()
    return app.exec_()
