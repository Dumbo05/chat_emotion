from pathlib import Path

import cv2
import numpy as np
import pytest

from emotion_app.recognizers.image import ImageRecognizer


@pytest.fixture
def image_recognizer() -> ImageRecognizer:
    recognizer = ImageRecognizer()
    if not recognizer.available:
        pytest.skip(f"local image model is not installed: {recognizer.status}")
    return recognizer


def test_image_recognizer_models_are_available(image_recognizer):
    assert image_recognizer.available
    assert "摄像头" in image_recognizer.status


def test_blank_image_returns_real_no_face_error(tmp_path, image_recognizer):
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.zeros((240, 320, 3), dtype=np.uint8))
    result = image_recognizer.predict(path)
    assert not result.ok
    assert "未检测到" in (result.error or "")
    assert result.probabilities == {}


def test_invalid_frame_is_rejected(image_recognizer):
    with pytest.raises(ValueError, match="尺寸过小"):
        image_recognizer.predict_frame(np.zeros((10, 10, 3), dtype=np.uint8))


def test_rotated_image_retries_orientation(image_recognizer):
    smoke_image = Path("cache/image-smoke-selfie.jpg")
    if not smoke_image.exists():
        pytest.skip("local smoke-test image is not installed")
    source = cv2.imread(str(smoke_image))
    assert source is not None
    rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
    faces = image_recognizer.predict_frame(rotated, try_rotations=True)
    assert faces
    assert faces[0].result.ok