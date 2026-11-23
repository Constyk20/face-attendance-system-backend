import os

os.environ.setdefault("FLASK_DEBUG", "1")

import numpy as np
import pytest

from services.face_recognition import (
    EMBEDDING_MODEL,
    FaceError,
    ModelMismatch,
    validate_embedding,
)


def unit(seed=0, dim=128):
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).round(6).tolist()


def test_valid_embedding_is_normalised_float32():
    out = validate_embedding(unit(1), EMBEDDING_MODEL)
    assert out.dtype == np.float32 and out.shape == (128,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "abc",
        [],
        [0.1] * 127,
        [0.1] * 129,
        [0.0] * 128,
        [0.5] * 128,  # constant, wrong norm
        [float("nan")] * 128,
        [float("inf")] + [0.0] * 127,
        ["a"] * 128,
        [True] * 128,
        [v * 5 for v in unit(2)],  # not unit length
        {"0": 1},
    ],
)
def test_malformed_embeddings_rejected(bad):
    with pytest.raises(FaceError):
        validate_embedding(bad, EMBEDDING_MODEL)


def test_constant_unit_vector_rejected():
    v = [1 / np.sqrt(128)] * 128
    with pytest.raises(FaceError):
        validate_embedding(v, EMBEDDING_MODEL)


@pytest.mark.parametrize("model", [None, "", "Facenet", "sface-yunet-v0"])
def test_other_models_rejected(model):
    with pytest.raises(ModelMismatch):
        validate_embedding(unit(3), model)
