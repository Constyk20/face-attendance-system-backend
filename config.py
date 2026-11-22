"""Central configuration. Every tunable lives here and can be set in .env."""

import os

from dotenv import load_dotenv

load_dotenv()

_INSECURE_DEFAULT = "dev-only-change-me"


def _get_bool(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name, default, minimum=None):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return value if minimum is None else max(minimum, value)


def _get_float(name, default, minimum=None):
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return value if minimum is None else max(minimum, value)


class Config:
    """Settings loaded from environment variables. Do not hard-code secrets."""

    # ----- Runtime -----------------------------------------------------
    DEBUG = _get_bool("FLASK_DEBUG", False)
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = _get_int("PORT", 5000, 1)
    # The Werkzeug reloader imports the app twice (two model loads). Off by default.
    USE_RELOADER = _get_bool("USE_RELOADER", False)
    WAITRESS_THREADS = _get_int("WAITRESS_THREADS", 4, 1)

    # ----- Security ----------------------------------------------------
    SECRET_KEY = os.getenv("SECRET_KEY", _INSECURE_DEFAULT)
    JWT_SECRET = os.getenv("JWT_SECRET", os.getenv("SECRET_KEY", _INSECURE_DEFAULT))
    JWT_EXPIRES_HOURS = _get_int("JWT_EXPIRES_HOURS", 8, 1)
    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    FRONTEND_ORIGINS = [
        item.strip() for item in FRONTEND_ORIGIN.split(",") if item.strip()
    ]

    # ----- MongoDB -----------------------------------------------------
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "face_attendance")
    MONGODB_MAX_POOL_SIZE = _get_int("MONGODB_MAX_POOL_SIZE", 20, 1)
    MONGODB_CONNECT_TIMEOUT_MS = _get_int("MONGODB_CONNECT_TIMEOUT_MS", 4000, 500)
    MONGODB_SERVER_SELECTION_TIMEOUT_MS = _get_int(
        "MONGODB_SERVER_SELECTION_TIMEOUT_MS", 4000, 500
    )
    MONGODB_SOCKET_TIMEOUT_MS = _get_int("MONGODB_SOCKET_TIMEOUT_MS", 20000, 1000)
    MONGODB_APP_NAME = os.getenv("MONGODB_APP_NAME", "face-attendance-api")

    # ----- Request / image limits -------------------------------------
    # Whole request body. Registration sends up to 8 images, so keep this generous.
    MAX_CONTENT_LENGTH = _get_int("MAX_CONTENT_LENGTH", 12 * 1024 * 1024, 1024)
    # One base64 image string (about 3 MB of base64 is roughly a 2.2 MB JPEG).
    MAX_IMAGE_B64_CHARS = _get_int("MAX_IMAGE_B64_CHARS", 3_000_000, 1000)
    MAX_IMAGE_DIMENSION = _get_int("MAX_IMAGE_DIMENSION", 4096, 64)
    MAX_IMAGE_PIXELS = _get_int("MAX_IMAGE_PIXELS", 16_000_000, 10_000)
    # Frames are downscaled to this longest side before face detection.
    DETECT_MAX_SIDE = _get_int("DETECT_MAX_SIDE", 640, 160)

    # ----- Face pipeline ----------------------------------------------
    PRELOAD_MODEL = _get_bool("PRELOAD_MODEL", True)
    # Facenet cosine distance. Lower means more similar. Calibrate before changing.
    FACE_DISTANCE_THRESHOLD = _get_float("FACE_DISTANCE_THRESHOLD", 0.40, 0.01)
    # Reject when the two best different students are closer than this.
    FACE_AMBIGUITY_MARGIN = _get_float("FACE_AMBIGUITY_MARGIN", 0.05, 0.0)
    FACE_MIN_SIZE = _get_int("FACE_MIN_SIZE", 80, 20)
    FACE_BLUR_MIN = _get_float("FACE_BLUR_MIN", 20.0, 0.0)
    FACE_MIN_BRIGHTNESS = _get_float("FACE_MIN_BRIGHTNESS", 25.0, 0.0)
    FACE_MAX_BRIGHTNESS = _get_float("FACE_MAX_BRIGHTNESS", 235.0, 1.0)
    FACE_EDGE_MARGIN_RATIO = _get_float("FACE_EDGE_MARGIN_RATIO", 0.01, 0.0)
    FACE_CROP_PADDING = _get_float("FACE_CROP_PADDING", 0.0, 0.0)
    FACE_ALIGN = _get_bool("FACE_ALIGN", True)
    # During recognition the largest face is used only if it is at least this many
    # times wider than the next face. Otherwise the frame is rejected as ambiguous.
    FACE_DOMINANCE_RATIO = _get_float("FACE_DOMINANCE_RATIO", 1.6, 1.0)

    # ----- Attendance / recognition -----------------------------------
    RECOGNITION_MIN_INTERVAL_MS = _get_int("RECOGNITION_MIN_INTERVAL_MS", 700, 0)
    # Frames waiting for the (single) inference slot across all sessions.
    MAX_PENDING_RECOGNITIONS = _get_int("MAX_PENDING_RECOGNITIONS", 4, 1)
    SESSION_CACHE_TTL_SECONDS = _get_float("SESSION_CACHE_TTL_SECONDS", 30.0, 0.0)
    FACE_CACHE_CHECK_INTERVAL_SECONDS = _get_float(
        "FACE_CACHE_CHECK_INTERVAL_SECONDS", 2.0, 0.0
    )
    # The frontend may still read "distance". Set to false once it does not.
    RETURN_DISTANCE = _get_bool("RETURN_DISTANCE", True)

    # ----- Reports -----------------------------------------------------
    EXPORT_MAX_ROWS = _get_int("EXPORT_MAX_ROWS", 20000, 100)

    # ----- Logging -----------------------------------------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE = os.getenv("LOG_FILE", "").strip()

    @classmethod
    def validate(cls):
        """Fail loudly on unsafe settings when not running in debug mode."""
        problems = []
        if not cls.DEBUG:
            if cls.SECRET_KEY == _INSECURE_DEFAULT:
                problems.append("SECRET_KEY is still the insecure default")
            if cls.JWT_SECRET == _INSECURE_DEFAULT:
                problems.append("JWT_SECRET is still the insecure default")
        if cls.FACE_MAX_BRIGHTNESS <= cls.FACE_MIN_BRIGHTNESS:
            problems.append("FACE_MAX_BRIGHTNESS must be greater than FACE_MIN_BRIGHTNESS")
        if problems:
            raise RuntimeError(
                "Unsafe configuration: "
                + "; ".join(problems)
                + ". Set real values in .env, or set FLASK_DEBUG=1 for local development."
            )
