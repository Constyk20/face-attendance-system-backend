"""Validation of face embeddings produced in the BROWSER.

Face detection and embedding (YuNet + SFace through ONNX Runtime Web) run on
the client. The server never sees a photo and loads no neural network: this
module only checks that what arrives is a well-formed embedding made by the
model this server was calibrated for.

Why "model" is checked explicitly: the previous DeepFace/Facenet embeddings and
the new SFace embeddings are BOTH 128 numbers long, so the length alone cannot
tell them apart. Comparing across models gives meaningless distances, so every
stored embedding and every request carries the model tag and only equal tags
are ever compared.
"""

import math

import numpy as np

from config import Config

EMBEDDING_MODEL = Config.EMBEDDING_MODEL  # must equal EMBEDDING_MODEL in faceEngine.js
EMBEDDING_DIM = Config.EMBEDDING_DIM
LEGACY_MODELS = ("Facenet",)  # embeddings from the old DeepFace pipeline
PIPELINE_VERSION = 3  # 1-2 = DeepFace/Facenet; 3 = browser YuNet + SFace

# The browser sends unit-length vectors. Allow rounding noise, reject anything else.
_NORM_MIN, _NORM_MAX = 0.95, 1.05


class FaceError(ValueError):
    """A problem with the submitted face data that is safe to show to the user."""


class ModelMismatch(FaceError):
    """The client used a different embedding model than this server expects."""


def check_model(model):
    if model != EMBEDDING_MODEL:
        raise ModelMismatch(
            "Your browser has an out-of-date face model. Reload the page (Ctrl+F5) and try again"
        )


def validate_embedding(value, model):
    """Return a unit-length float32 vector of EMBEDDING_DIM numbers, or raise FaceError."""
    check_model(model)  # the tag is mandatory
    if not isinstance(value, (list, tuple)) or len(value) != EMBEDDING_DIM:
        raise FaceError("Face data is not valid")
    try:
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise FaceError("Face data is not valid")
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise FaceError("Face data is not valid") from None
    if not np.all(np.isfinite(vector)) or np.abs(vector).max() > 10:
        raise FaceError("Face data is not valid")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or not (_NORM_MIN <= norm <= _NORM_MAX):
        raise FaceError("Face data is not valid")
    if float(vector.std()) < 1e-3:  # constant vector: not a face
        raise FaceError("Face data is not valid")
    return (vector / norm).astype(np.float32)


def model_status():
    """Reported by /api/health. The model runs in the browser, not here."""
    return "browser"
