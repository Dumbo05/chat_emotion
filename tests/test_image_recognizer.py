import cv2
import numpy as np

from emotion_app.recognizers.image import ImageRecognizer


def test_image_recognizer_models_are_available():
    recognizer = ImageRecognizer()
    assert recognizer.available
    assert "摄像头" in recognizer.status


def test_blank_image_returns_real_no_face_error(tmp_path):
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.zeros((240, 320, 3), dtype=np.uint8))
    result = ImageRecognizer().predict(path)
    assert not result.ok
    assert "未检测到" in (result.error or "")
    assert result.probabilities == {}


def test_invalid_frame_is_rejected():
    recognizer = ImageRecognizer()
    try:
        recognizer.predict_frame(np.zeros((10, 10, 3), dtype=np.uint8))
    except ValueError as exc:
        assert "尺寸过小" in str(exc)
    else:
        raise AssertionError("small frame should be rejected")


def test_rotated_image_retries_orientation():
    source = cv2.imread("cache/image-smoke-selfie.jpg")
    assert source is not None
    rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
    faces = ImageRecognizer().predict_frame(rotated, try_rotations=True)
    assert faces
    assert faces[0].result.ok
