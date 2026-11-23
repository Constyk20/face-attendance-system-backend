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

    # ----- Request limits -----------------------------------------------
    # Face data is a handful of 128-number vectors, so requests are tiny.
    MAX_CONTENT_LENGTH = _get_int("MAX_CONTENT_LENGTH", 256 * 1024, 1024)

    # ----- Face matching -------------------------------------------------
    # Faces are detected and embedded IN THE BROWSER (YuNet + SFace, ONNX Runtime
    # Web). This tag is stored with every embedding and must match the frontend's
    # EMBEDDING_MODEL. Only embeddings with this tag are ever compared.
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sface-yunet-v1").strip()
    EMBEDDING_DIM = 128
    # SFace cosine DISTANCE (1 - cosine similarity). Lower = more similar.
    # Measured with the browser pipeline on 300 labelled pairs: same person <= 0.40,
    # different people >= 0.66. 0.50 sits between them; OpenCV's published SFace
    # limit is 0.637 (similarity 0.363) and is deliberately NOT used because a
    # school attendance system should prefer a missed face over a wrong student.
    # The old Facenet value (0.40) must not be reused: the scales differ.
    FACE_DISTANCE_THRESHOLD = _get_float("FACE_DISTANCE_THRESHOLD", 0.50, 0.01)
    # Reject when the two best different students are closer than this.
    FACE_AMBIGUITY_MARGIN = _get_float("FACE_AMBIGUITY_MARGIN", 0.05, 0.0)

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
        if not cls.EMBEDDING_MODEL:
            problems.append("EMBEDDING_MODEL must not be empty")
        if problems:
            raise RuntimeError(
                "Unsafe configuration: "
                + "; ".join(problems)
                + ". Set real values in .env, or set FLASK_DEBUG=1 for local development."
            )
