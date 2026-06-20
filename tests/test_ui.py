from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton

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

class FakeSpeechRecognizer:
    available = True
    status = "语音测试模型已加载"
    metrics = {
        "test": {"accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0, "f1_macro": 1.0},
        "speaker_holdout": {"average": {"accuracy": .4775, "precision_macro": .5207,
                                           "recall_macro": .4778, "f1_macro": .4016}},
    }

    def predict(self, path):
        return RecognitionResult.failure("not used")


def test_speech_tab_hides_model_evaluation_metrics():
    get_app()
    window = MainWindow(FakeTextRecognizer(), FakeSpeechRecognizer(), ImageRecognizer())
    labels = [label.text() for label in window.tabs.widget(1).findChildren(type(window.speech_result_label))]
    assert not any("准确率" in text or "Macro-F1" in text for text in labels)
    assert "等待识别" in labels
    window.close()

def test_speech_picker_labels_and_filters_wav_and_mp3(monkeypatch):
    get_app()
    captured = {}

    def fake_picker(parent, title, directory, file_filter):
        captured["filter"] = file_filter
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_picker)
    window = MainWindow(FakeTextRecognizer(), FakeSpeechRecognizer(), ImageRecognizer())
    buttons = window.tabs.widget(1).findChildren(QPushButton)
    picker = next(button for button in buttons if "WAV / MP3" in button.text())
    picker.click()

    assert picker.text() == "选择 WAV / MP3 音频"
    assert captured["filter"] == "全部音频 (*.wav *.mp3);;WAV 音频 (*.wav);;MP3 音频 (*.mp3)"
    window.close()
