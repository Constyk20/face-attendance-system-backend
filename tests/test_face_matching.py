import numpy as np
import pytest

from services.face_matching import (
    build_snapshot,
    conflicts_with_others,
    consistency_outliers,
    match,
    normalize,
)

DIM = 128
THRESHOLD = 0.40
MARGIN = 0.05


def unit(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=DIM).astype(np.float32)


def near(vector, distance, seed=99):
    """A vector whose cosine distance from `vector` is about `distance`."""
    v = normalize(vector)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=DIM).astype(np.float32)
    noise -= noise.dot(v) * v
    noise /= np.linalg.norm(noise)
    cos = 1.0 - distance
    return cos * v + np.sqrt(1 - cos**2) * noise


def snap(people):
    items = []
    for sid, samples in people.items():
        for s in samples:
            items.append((sid, {"id": sid}, s))
    return build_snapshot(items)


def test_match_known_student():
    a, b = unit(1), unit(2)
    s = snap({"a": [a, near(a, 0.1)], "b": [b]})
    result = match(s, near(a, 0.15, seed=5), THRESHOLD, MARGIN)
    assert result.status == "match"
    assert s.student_ids[result.index] == "a"
    assert result.distance < 0.2


def test_unknown_face_rejected():
    s = snap({"a": [unit(1)], "b": [unit(2)]})
    assert match(s, unit(3), THRESHOLD, MARGIN).status == "unknown"


def test_threshold_is_exclusive_and_not_loosened():
    a = unit(1)
    s = snap({"a": [a]})
    assert match(s, near(a, 0.39), THRESHOLD, MARGIN).status == "match"
    assert match(s, near(a, 0.45), THRESHOLD, MARGIN).status == "unknown"


def test_ambiguous_when_two_students_are_close():
    a = unit(1)
    s = snap({"a": [near(a, 0.20, seed=3)], "b": [near(a, 0.22, seed=4)]})
    assert match(s, a, THRESHOLD, MARGIN).status == "ambiguous"


def test_same_student_many_samples_is_not_ambiguous():
    a = unit(1)
    s = snap({"a": [near(a, 0.10, seed=i) for i in range(5)], "b": [unit(2)]})
    assert match(s, a, THRESHOLD, MARGIN).status == "match"


def test_single_student_has_no_runner_up():
    a = unit(1)
    s = snap({"a": [a]})
    result = match(s, a, THRESHOLD, MARGIN)
    assert result.status == "match" and result.second_distance is None


def test_empty_snapshot():
    s = build_snapshot([])
    assert match(s, unit(1), THRESHOLD, MARGIN).status == "empty"


def test_wrong_dimension_is_invalid():
    s = snap({"a": [unit(1)]})
    assert match(s, np.ones(64, dtype=np.float32), THRESHOLD, MARGIN).status == "invalid"


def test_bad_records_are_skipped_not_fatal():
    items = [
        ("a", {}, unit(1)),
        ("b", {}, []),  # empty
        ("c", {}, [float("nan")] * DIM),  # NaN
        ("d", {}, [0.0] * DIM),  # zero vector
        ("e", {}, np.ones(64)),  # different model / size
    ]
    s = build_snapshot(items)
    assert s.student_ids == ("a",)
    assert s.skipped == 4


def test_unnormalised_legacy_embeddings_are_handled():
    a = unit(1)
    s = snap({"a": [a * 37.0]})  # old records were not normalised
    assert match(s, a, THRESHOLD, MARGIN).status == "match"


def test_registration_conflict_detection():
    a, b = unit(1), unit(2)
    s = snap({"a": [a], "b": [b]})
    assert conflicts_with_others(s, [near(a, 0.1)], "b", THRESHOLD)
    assert not conflicts_with_others(s, [near(a, 0.1)], "a", THRESHOLD)  # own face
    assert not conflicts_with_others(s, [unit(3)], "b", THRESHOLD)


def test_consistency_outliers():
    a = unit(1)
    good = [near(a, 0.08, seed=i) for i in range(4)]
    assert consistency_outliers(good, THRESHOLD) == []
    assert consistency_outliers(good + [unit(50)], THRESHOLD) == [4]


def test_cache_excludes_inactive_students_via_builder():
    """Inactive students are filtered out BEFORE the matrix is built."""
    a = unit(1)
    active = {"a": True, "b": False}
    items = [
        (sid, {}, vec)
        for sid, vec in (("a", unit(7)), ("b", a))
        if active[sid]
    ]
    s = build_snapshot(items)
    # b (inactive) is the true match for `a`, but cannot win or hide anyone.
    assert match(s, a, THRESHOLD, MARGIN).status == "unknown"
