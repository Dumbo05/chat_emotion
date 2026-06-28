from emotion_app.domain import EMOTIONS
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


class FakeSpeechProbabilityModel:
    classes_ = list(EMOTIONS)

    def predict_proba(self, features):
        return [[0.01, 0.01, 0.01, 0.90, 0.02, 0.03, 0.02]]


def test_trained_speech_recognizer_returns_seven_probabilities(tmp_path):
    import wave

    from emotion_app.recognizers.speech import SpeechRecognizer as TrainedSpeechRecognizer

    model_dir = tmp_path / "speech"
    model_dir.mkdir()
    (model_dir / "speech_model.joblib").write_bytes(b"test stub")
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 1600)

    recognizer = TrainedSpeechRecognizer(model_dir)
    recognizer._model = FakeSpeechProbabilityModel()
    result = recognizer.predict(audio)

    assert result.ok
    assert result.emotion == "joy"
    assert result.confidence == 0.90
    assert set(result.probabilities) == set(EMOTIONS)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9
