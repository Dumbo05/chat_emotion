from emotion_app.recognizers.placeholders import ImageRecognizer, SpeechRecognizer
from emotion_app.recognizers.text import TextRecognizer


def test_missing_text_model_returns_error_not_prediction(tmp_path):
    recognizer = TextRecognizer(tmp_path / "missing")
    result = recognizer.predict("I am happy")
    assert not result.ok
    assert result.emotion is None
    assert result.probabilities == {}
    assert "未找到文本模型" in (result.error or "")


def test_empty_text_is_rejected_before_model_loading(tmp_path):
    result = TextRecognizer(tmp_path / "missing").predict("   ")
    assert not result.ok
    assert result.error == "文本内容不能为空"


def test_future_modalities_never_fake_results(tmp_path):
    audio = tmp_path / "sample.wav"
    image = tmp_path / "sample.png"
    audio.write_bytes(b"RIFF")
    image.write_bytes(b"PNG")
    for recognizer, path in ((SpeechRecognizer(), audio), (ImageRecognizer(), image)):
        assert not recognizer.available
        result = recognizer.predict(path)
        assert not result.ok
        assert result.probabilities == {}

