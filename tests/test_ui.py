from PyQt5.QtWidgets import QApplication

from emotion_app.domain import EMOTIONS, RecognitionResult
from emotion_app.recognizers.placeholders import ImageRecognizer, SpeechRecognizer
from emotion_app.ui.main_window import MainWindow


class FakeTextRecognizer:
    available = True
    status = "测试模型已加载"

    def predict(self, text: str) -> RecognitionResult:
        probabilities = {emotion: 0.0 for emotion in EMOTIONS}
        probabilities["joy"] = 1.0
        return RecognitionResult("joy", 1.0, probabilities, "fake")


_APP = None


def get_app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_main_window_has_three_modality_tabs():
    get_app()
    window = MainWindow(FakeTextRecognizer(), SpeechRecognizer(), ImageRecognizer())
    assert window.tabs.count() == 3
    assert [window.tabs.tabText(index) for index in range(3)] == ["文本识别", "语音识别", "图像识别"]
    window.close()


def test_result_renders_all_probabilities():
    get_app()
    window = MainWindow(FakeTextRecognizer(), SpeechRecognizer(), ImageRecognizer())
    result = FakeTextRecognizer().predict("hello")
    window._show_result(result)
    assert window.result_label.text() == "喜悦"
    assert window.confidence_label.text() == "置信度：100.00%"
    assert window._probability_bars["joy"].value() == 1000
    window.close()
