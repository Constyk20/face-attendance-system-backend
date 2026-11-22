"""Pure NumPy face matching. No database, Flask or DeepFace imports.

Embeddings are L2-normalised once, so cosine distance for ALL stored samples is
one matrix-vector product:  distance = 1 - matrix @ query
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def normalize(vector):
    """Return a unit-length float32 vector, or None if it is empty/invalid/zero."""
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if norm < _EPS:
        return None
    return array / norm


@dataclass(eq=False)
class Snapshot:
    """Immutable-by-convention view of all enrolled samples.

    Rows of `matrix` are grouped by student, so per-student minimums are one
    np.minimum.reduceat call.
    """

    matrix: np.ndarray  # (N, D) float32, rows unit length
    offsets: np.ndarray  # (S,) first row of each student
    owner_row: np.ndarray  # (N,) student index of each row
    student_ids: tuple  # (S,)
    labels: tuple  # (S,) whatever the caller attached (name, matric, ...)
    version: object = None
    skipped: int = 0

    @property
    def sample_count(self):
        return int(self.matrix.shape[0])

    @property
    def student_count(self):
        return len(self.student_ids)

    @property
    def dim(self):
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    def index_of(self, student_id):
        try:
            return self.student_ids.index(student_id)
        except ValueError:
            return None


def empty_snapshot(version=None, skipped=0):
    return Snapshot(
        matrix=np.empty((0, 0), dtype=np.float32),
        offsets=np.empty(0, dtype=np.intp),
        owner_row=np.empty(0, dtype=np.int32),
        student_ids=(),
        labels=(),
        version=version,
        skipped=skipped,
    )


def build_snapshot(items, version=None):
    """Build a Snapshot from an iterable of (student_id, label, embedding).

    Tolerant of old records: embeddings are normalised here, empty/NaN/zero
    vectors are skipped, and if records of different sizes exist (for example
    after a model change) only the most common size is kept.
    """
    vectors = []
    skipped = 0
    for student_id, label, embedding in items:
        vector = normalize(embedding)
        if vector is None:
            skipped += 1
            continue
        vectors.append((student_id, label, vector))

    if not vectors:
        return empty_snapshot(version, skipped)

    dim = Counter(vec.shape[0] for _, _, vec in vectors).most_common(1)[0][0]
    grouped = {}
    for student_id, label, vector in vectors:
        if vector.shape[0] != dim:
            skipped += 1
            continue
        entry = grouped.get(student_id)
        if entry is None:
            grouped[student_id] = [label, [vector]]
        else:
            entry[1].append(vector)

    total = sum(len(entry[1]) for entry in grouped.values())
    matrix = np.empty((total, dim), dtype=np.float32)
    offsets = np.empty(len(grouped), dtype=np.intp)
    owner_row = np.empty(total, dtype=np.int32)
    row = 0
    for index, (_, (_, vecs)) in enumerate(grouped.items()):
        offsets[index] = row
        for vec in vecs:
            matrix[row] = vec
            owner_row[row] = index
            row += 1

    return Snapshot(
        matrix=matrix,
        offsets=offsets,
        owner_row=owner_row,
        student_ids=tuple(grouped.keys()),
        labels=tuple(entry[0] for entry in grouped.values()),
        version=version,
        skipped=skipped,
    )


@dataclass(frozen=True)
class MatchResult:
    status: str  # "match" | "unknown" | "ambiguous" | "empty" | "invalid"
    index: int = None  # student index into Snapshot.labels (match only)
    distance: float = None  # best per-student distance
    second_distance: float = None  # best distance of the runner-up student


def match(snapshot, query, threshold, margin):
    """Identify one query embedding.

    - distance per student = smallest distance over that student's samples
    - unknown   : best distance >= threshold
    - ambiguous : a DIFFERENT student is within `margin` of the best one
    - match     : otherwise
    """
    if snapshot.sample_count == 0:
        return MatchResult("empty")
    vector = normalize(query)
    if vector is None or vector.shape[0] != snapshot.dim:
        return MatchResult("invalid")

    distances = 1.0 - snapshot.matrix @ vector  # (N,)
    per_student = np.minimum.reduceat(distances, snapshot.offsets)  # (S,)

    best_index = int(np.argmin(per_student))
    best = float(per_student[best_index])
    second = None
    if per_student.size > 1:
        second = float(np.partition(per_student, 1)[1])

    if best >= threshold:
        return MatchResult("unknown", None, best, second)
    if second is not None and (second - best) < margin:
        return MatchResult("ambiguous", None, best, second)
    return MatchResult("match", best_index, best, second)


def conflicts_with_others(snapshot, embeddings, exclude_student_id, threshold):
    """True if any of `embeddings` is within `threshold` of another student's sample."""
    if snapshot.sample_count == 0:
        return False
    queries = [normalize(e) for e in embeddings]
    queries = [q for q in queries if q is not None and q.shape[0] == snapshot.dim]
    if not queries:
        return False

    distances = 1.0 - snapshot.matrix @ np.stack(queries, axis=1)  # (N, m)
    keep = np.ones(snapshot.sample_count, dtype=bool)
    own = snapshot.index_of(exclude_student_id)
    if own is not None:
        keep &= snapshot.owner_row != own
    if not keep.any():
        return False
    return bool((distances[keep] < threshold).any())


def consistency_outliers(embeddings, threshold):
    """Indexes of embeddings that do not resemble the rest of the set.

    An embedding is an outlier when its median distance to the others is at or
    above the same-person threshold.
    """
    vectors = [normalize(e) for e in embeddings]
    if len(vectors) < 3 or any(v is None for v in vectors):
        return list(range(len(vectors))) if any(v is None for v in vectors) else []
    matrix = np.stack(vectors)
    distances = 1.0 - matrix @ matrix.T
    np.fill_diagonal(distances, np.nan)
    medians = np.nanmedian(distances, axis=1)
    return [i for i, value in enumerate(medians) if value >= threshold]
