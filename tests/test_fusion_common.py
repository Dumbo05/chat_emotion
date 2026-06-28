import numpy as np

from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import (
    array_to_probs,
    evaluate_config,
    gated_fusion,
    normalize_probs,
    weighted_fusion,
)


def test_normalize_probs_returns_full_label_space():
    probs = normalize_probs({"happy": 2, "anger": 1})

    assert set(probs) == set(EMOTIONS)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["joy"] > probs["anger"]


def test_weighted_fusion_skips_missing_modalities():
    row = {
        "label": "joy",
        "image_ok": False,
        "speech_ok": True,
        "text_ok": True,
        "speech_probs": array_to_probs(np.eye(len(EMOTIONS))[EMOTIONS.index("anger")]),
        "text_probs": array_to_probs(np.eye(len(EMOTIONS))[EMOTIONS.index("joy")]),
    }
    config = {
        "method": "weighted",
        "weights": {"image": 0.8, "speech": 0.1, "text": 0.1},
        "temperatures": {"image": 1.0, "speech": 1.0, "text": 1.0},
    }

    fused = weighted_fusion(row, config)

    assert abs(float(fused.sum()) - 1.0) < 1e-9
    assert EMOTIONS[int(fused.argmax())] in {"anger", "joy"}


def test_gated_fusion_lets_high_confidence_image_dominate():
    row = {
        "label": "sadness",
        "image_ok": True,
        "speech_ok": True,
        "text_ok": True,
        "image_probs": array_to_probs(np.eye(len(EMOTIONS))[EMOTIONS.index("sadness")]),
        "speech_probs": array_to_probs(np.eye(len(EMOTIONS))[EMOTIONS.index("joy")]),
        "text_probs": array_to_probs(np.eye(len(EMOTIONS))[EMOTIONS.index("joy")]),
    }
    config = {
        "method": "gated",
        "weights": {"image": 0.4, "speech": 0.3, "text": 0.3},
        "temperatures": {"image": 1.0, "speech": 1.0, "text": 1.0},
        "gate": {
            "image_confidence_threshold": 0.75,
            "image_high_confidence_weights": {"image": 0.9, "speech": 0.05, "text": 0.05},
            "image_missing_weights": {"image": 0.0, "speech": 0.7, "text": 0.3},
        },
    }

    fused = gated_fusion(row, config)

    assert EMOTIONS[int(fused.argmax())] == "sadness"


def test_evaluate_config_reports_accuracy():
    row = {
        "label": "joy",
        "image_ok": True,
        "speech_ok": False,
        "text_ok": False,
        "image_probs": array_to_probs(np.eye(len(EMOTIONS))[EMOTIONS.index("joy")]),
    }
    config = {
        "method": "weighted",
        "weights": {"image": 1.0, "speech": 0.0, "text": 0.0},
        "temperatures": {"image": 1.0, "speech": 1.0, "text": 1.0},
    }

    metrics = evaluate_config([row], config)

    assert metrics["accuracy"] == 1.0
