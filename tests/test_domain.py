import pytest

from emotion_app.domain import EMOTIONS, RecognitionResult, normalize_emotion


def test_aliases_normalize_to_canonical_labels():
    assert normalize_emotion("happy") == "joy"
    assert normalize_emotion("sad") == "sadness"
    assert normalize_emotion("angry") == "anger"


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        normalize_emotion("made-up")


def test_result_success_and_failure_are_explicit():
    probabilities = {label: 0.0 for label in EMOTIONS}
    probabilities["joy"] = 1.0
    success = RecognitionResult("joy", 1.0, probabilities, "fake")
    failure = RecognitionResult.failure("missing model")
    assert success.ok
    assert success.to_dict()["emotion_zh"] == "喜悦"
    assert not failure.ok
    assert failure.error == "missing model"
