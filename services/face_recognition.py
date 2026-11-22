"""Detect ONE face, quality-check it and turn it into a Facenet embedding.

Pipeline (identical for registration and recognition):

  base64 -> size/dimension checks -> decode (once)
         -> downscale copy -> Haar detection (once, thread-local detector)
         -> pick face -> quality gate -> crop from the ORIGINAL frame
         -> optional eye-level alignment -> 160x160
         -> DeepFace.represent(detector_backend="skip")  (no second detection)
         -> L2-normalised float32 vector

Raw image bytes are used only in memory. Nothing is written to disk.
DeepFace/TensorFlow is imported lazily, so `python app.py` starts instantly and
the model is loaded once, in the background, by start_warmup().
"""

import base64
import binascii
import logging
import math
import struct
import threading

import cv2
import numpy as np

from config import Config
from services.face_matching import normalize

log = logging.getLogger(__name__)

MODEL_NAME = "Facenet"
PIPELINE_VERSION = 2  # stored with every new embedding
MODEL_INPUT_SIZE = (160, 160)

# Kept for code that still imports the old names.
SAME_PERSON_DISTANCE = Config.FACE_DISTANCE_THRESHOLD


class FaceError(ValueError):
    """A problem with the submitted image that is safe to show to the user."""


class ModelLoading(RuntimeError):
    """The face model is still being loaded by another thread."""


# --------------------------------------------------------------------------
# Image decoding
# --------------------------------------------------------------------------
def _peek_dimensions(raw):
    """Read (width, height) from a JPEG/PNG header without decoding the pixels."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        width, height = struct.unpack(">II", raw[16:24])
        return width, height
    if raw[:2] == b"\xff\xd8":
        i, n = 2, len(raw)
        while i + 4 <= n:
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if marker in (0xD9, 0xDA):
                return None
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if i + 9 <= n:
                    height, width = struct.unpack(">HH", raw[i + 5 : i + 9])
                    return width, height
                return None
            i += 2 + struct.unpack(">H", raw[i + 2 : i + 4])[0]
    return None


def _check_dimensions(width, height):
    if width <= 0 or height <= 0:
        raise FaceError("Image could not be decoded")
    if max(width, height) > Config.MAX_IMAGE_DIMENSION:
        raise FaceError("Image dimensions are too large")
    if width * height > Config.MAX_IMAGE_PIXELS:
        raise FaceError("Image dimensions are too large")


def decode_image(image_value):
    """Accept a raw base64 string or a data-URL. Return a BGR image."""
    if not isinstance(image_value, str):
        raise FaceError("Image is not valid base64")
    text = image_value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    if len(text) > Config.MAX_IMAGE_B64_CHARS:
        raise FaceError("Image is too large")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FaceError("Image is not valid base64") from exc
    if not raw:
        raise FaceError("Image is empty")

    # Reject decompression bombs BEFORE allocating the pixel buffer.
    header = _peek_dimensions(raw)
    if header:
        _check_dimensions(*header)

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FaceError("Image could not be decoded")
    height, width = image.shape[:2]
    _check_dimensions(width, height)
    return image


# --------------------------------------------------------------------------
# Detection, quality gate, alignment
# --------------------------------------------------------------------------
_tls = threading.local()


def _detectors():
    """Haar cascades are not guaranteed thread-safe, so each thread gets its own."""
    face = getattr(_tls, "face", None)
    if face is None:
        base = cv2.data.haarcascades
        face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        if face.empty() or eye.empty():
            raise RuntimeError("OpenCV Haar cascade files could not be loaded")
        _tls.face, _tls.eye = face, eye
    return _tls.face, _tls.eye


def _detect_faces(image):
    """Return [(x, y, w, h)] in ORIGINAL image coordinates."""
    height, width = image.shape[:2]
    scale = min(1.0, Config.DETECT_MAX_SIDE / float(max(height, width)))
    small = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    min_px = max(24, int(round(Config.FACE_MIN_SIZE * scale)))
    face_detector, _ = _detectors()
    found = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_px, min_px)
    )
    boxes = []
    for x, y, w, h in (found if len(found) else []):
        x0, y0 = int(x / scale), int(y / scale)
        x1, y1 = min(width, int((x + w) / scale)), min(height, int((y + h) / scale))
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def _select_face(boxes, policy):
    """policy "single": exactly one face (registration).
    policy "dominant": the largest face, if it clearly dominates the frame."""
    if policy not in ("single", "dominant"):
        raise ValueError("unknown face policy")
    if not boxes:
        raise FaceError("No face detected")
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    if len(boxes) > 1:
        if policy == "single" or boxes[0][2] < Config.FACE_DOMINANCE_RATIO * boxes[1][2]:
            raise FaceError("More than one face detected")
    return boxes[0]


def _check_geometry(box, width, height):
    x, y, w, h = box
    if min(w, h) < Config.FACE_MIN_SIZE:
        raise FaceError("Face is too small. Move closer to the camera")
    margin = int(round(Config.FACE_EDGE_MARGIN_RATIO * min(width, height)))
    if x < margin or y < margin or x + w > width - margin or y + h > height - margin:
        raise FaceError("Face is too close to the edge of the frame")


def _check_quality(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < Config.FACE_MIN_BRIGHTNESS or brightness > Config.FACE_MAX_BRIGHTNESS:
        raise FaceError("Lighting is poor. Face the light and try again")
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < Config.FACE_BLUR_MIN:
        raise FaceError("Image is too blurry. Hold still and try again")


def _align(crop):
    """Rotate the crop so the eyes are level. Returns the crop unchanged if unsure."""
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, eye_detector = _detectors()
    min_eye = max(8, width // 10)
    eyes = eye_detector.detectMultiScale(
        gray[: int(height * 0.6)], scaleFactor=1.1, minNeighbors=5, minSize=(min_eye, min_eye)
    )
    if len(eyes) < 2:
        return crop
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
    dx = (x2 + w2 / 2.0) - (x1 + w1 / 2.0)
    dy = (y2 + h2 / 2.0) - (y1 + h1 / 2.0)
    if abs(dx) < width * 0.15:  # both boxes on the same eye: not a real pair
        return crop
    angle = math.degrees(math.atan2(dy, dx))
    if abs(angle) > 30:
        return crop
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        crop, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def prepare_face(image, policy="single"):
    """Return a 160x160 BGR face crop ready for the model, or raise FaceError."""
    height, width = image.shape[:2]
    box = _select_face(_detect_faces(image), policy)
    _check_geometry(box, width, height)

    x, y, w, h = box
    pad = int(round(Config.FACE_CROP_PADDING * max(w, h)))
    crop = image[
        max(0, y - pad) : min(height, y + h + pad),
        max(0, x - pad) : min(width, x + w + pad),
    ]
    if Config.FACE_ALIGN:
        crop = _align(crop)
    interpolation = cv2.INTER_AREA if crop.shape[0] > MODEL_INPUT_SIZE[1] else cv2.INTER_LINEAR
    face = cv2.resize(crop, MODEL_INPUT_SIZE, interpolation=interpolation)
    _check_quality(face)
    return face


# --------------------------------------------------------------------------
# Model (loaded once, inference serialised)
# --------------------------------------------------------------------------
_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_deepface = None
_model_state = "not_loaded"  # not_loaded | loading | ready | error


def _ensure_model():
    """Load the model once. Never blocks behind another thread's load."""
    global _deepface, _model_state
    if _model_state == "ready":
        return
    if not _load_lock.acquire(blocking=False):
        raise ModelLoading("Face engine is starting. Try again in a few seconds.")
    try:
        if _model_state == "ready":
            return
        _model_state = "loading"
        try:
            from deepface import DeepFace  # heavy import, done once, lazily

            dummy = np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype=np.uint8)
            with _infer_lock:
                DeepFace.represent(
                    img_path=dummy,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )
            _deepface = DeepFace
            _model_state = "ready"
            log.info("Face model %s loaded and warmed up", MODEL_NAME)
        except Exception:
            _model_state = "error"
            log.exception("Face model could not be loaded")
            raise RuntimeError("Face model could not be loaded")
    finally:
        _load_lock.release()


def start_warmup():
    """Load the model in a background thread so server start-up is not delayed."""

    def target():
        try:
            _ensure_model()
        except Exception:
            log.warning("Warm-up failed; the model will be retried on first use")

    threading.Thread(target=target, name="face-model-warmup", daemon=True).start()


def model_status():
    return _model_state


def extract_embedding(image, policy="single"):
    """Return an L2-normalised float32 embedding (np.ndarray) for one face."""
    face = prepare_face(image, policy)
    _ensure_model()
    with _infer_lock:  # TensorFlow inference is serialised
        results = _deepface.represent(
            img_path=face,
            model_name=MODEL_NAME,
            detector_backend="skip",  # the face is already cropped: no 2nd detection
            enforce_detection=False,
            align=False,
        )
    if isinstance(results, dict):
        results = [results]
    vector = normalize(results[0].get("embedding")) if results else None
    if vector is None:
        raise FaceError("Face embedding could not be created")
    return vector


def cosine_distance(left, right):
    """Legacy helper kept for compatibility. New code uses face_matching."""
    a, b = normalize(left), normalize(right)
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(1.0 - np.dot(a, b))
