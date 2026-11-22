import base64
import os

os.environ.setdefault("FLASK_DEBUG", "1")

import cv2
import numpy as np
import pytest

from services.face_recognition import FaceError, decode_image


def b64_jpeg(width, height):
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_valid_image_and_data_url():
    data = b64_jpeg(200, 150)
    assert decode_image(data).shape[:2] == (150, 200)
    assert decode_image("data:image/jpeg;base64," + data).shape[:2] == (150, 200)


@pytest.mark.parametrize("value", [None, 123, "", "not base64!!", "data:image/jpeg;base64,"])
def test_garbage_is_rejected(value):
    with pytest.raises(FaceError):
        decode_image(value)


def test_oversized_dimensions_rejected_before_decode():
    with pytest.raises(FaceError, match="too large"):
        decode_image(b64_jpeg(5000, 100))


def test_oversized_payload_rejected(monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "MAX_IMAGE_B64_CHARS", 100)
    with pytest.raises(FaceError, match="too large"):
        decode_image(b64_jpeg(300, 300))
