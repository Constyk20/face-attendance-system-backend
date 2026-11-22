"""Shared in-memory index of enrolled face embeddings.

Why this exists: the old code read EVERY embedding from MongoDB and looped over
them in Python for every webcam frame. Now they are loaded once into a NumPy
matrix and refreshed only when they change.

Staleness protection:
  * invalidate() is called by every route that changes faces or the student
    fields kept in the cache (name, matric number, active flag). It bumps a
    version counter in MongoDB and forces an immediate rebuild in this process.
  * Other processes (or manual edits) are detected by a cheap check, at most
    every FACE_CACHE_CHECK_INTERVAL_SECONDS: the version counter plus the
    collection's estimated document count (an O(1) metadata read).
  * Only ACTIVE students are loaded, so an inactive student can never win a
    match and hide the correct active one.
"""

import logging
import threading
import time
from datetime import datetime, timezone

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from config import Config
from database.db import get_db
from services.face_matching import (
    Snapshot,
    build_snapshot,
    conflicts_with_others,
    empty_snapshot,
)
from services.face_matching import match as match_snapshot

log = logging.getLogger(__name__)

_META_ID = "face_data_version"


def read_fingerprint():
    """(version counter, estimated embedding count). Two cheap reads."""
    db = get_db()
    doc = db.meta.find_one({"_id": _META_ID}, {"value": 1})
    version = int(doc["value"]) if doc and "value" in doc else 0
    return version, int(db.face_data.estimated_document_count())


def bump_version():
    doc = get_db().meta.find_one_and_update(
        {"_id": _META_ID},
        {"$inc": {"value": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
        projection={"value": 1},
    )
    return int(doc["value"])


class FaceIndex:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = empty_snapshot()
        self._loaded = False
        self._dirty = True
        self._last_check = 0.0
        self._built_at = None
        self._legacy_samples = 0

    # ----- reading -----------------------------------------------------
    def snapshot(self, force=False):
        """Return an up-to-date Snapshot (rebuilding only when needed)."""
        if not force and self._loaded and not self._dirty and self._recently_checked():
            return self._snapshot
        with self._lock:
            if (
                not force
                and self._loaded
                and not self._dirty
                and self._recently_checked()
            ):
                return self._snapshot
            fingerprint = read_fingerprint()
            self._last_check = time.monotonic()
            if (
                force
                or not self._loaded
                or self._dirty
                or fingerprint != self._snapshot.version
            ):
                # Clear the flag BEFORE reading so an invalidate() that happens
                # while building leaves it set for the next call.
                self._dirty = False
                self._snapshot = self._build(fingerprint)
                self._loaded = True
            return self._snapshot

    def _recently_checked(self):
        return (
            time.monotonic() - self._last_check
            < Config.FACE_CACHE_CHECK_INTERVAL_SECONDS
        )

    def _build(self, fingerprint):
        started = time.perf_counter()
        db = get_db()
        labels = {
            student["_id"]: {
                "id": student["_id"],
                "full_name": student.get("full_name", ""),
                "matric_number": student.get("matric_number", ""),
            }
            for student in db.students.find(
                {"is_active": True}, {"full_name": 1, "matric_number": 1}
            )
        }

        legacy = 0

        def items():
            nonlocal legacy
            cursor = db.face_data.find(
                {}, {"student_id": 1, "embedding": 1, "pipeline_version": 1}
            ).batch_size(1000)
            for doc in cursor:
                label = labels.get(doc.get("student_id"))
                if label is None:  # inactive or deleted student
                    continue
                if not doc.get("pipeline_version"):
                    legacy += 1
                yield doc["student_id"], label, doc.get("embedding")

        snapshot = build_snapshot(items(), version=fingerprint)
        self._legacy_samples = legacy
        self._built_at = datetime.now(timezone.utc)
        log.info(
            "Face index rebuilt: %d students, %d samples, %d skipped in %.0f ms",
            snapshot.student_count,
            snapshot.sample_count,
            snapshot.skipped,
            (time.perf_counter() - started) * 1000,
        )
        return snapshot

    # ----- changing ----------------------------------------------------
    def invalidate(self):
        """Call after faces or cached student fields change."""
        try:
            bump_version()
        except PyMongoError:
            log.exception("Could not bump the face data version")
        self._dirty = True

    # ----- using -------------------------------------------------------
    def match(self, embedding):
        snapshot = self.snapshot()
        return snapshot, match_snapshot(
            snapshot,
            embedding,
            Config.FACE_DISTANCE_THRESHOLD,
            Config.FACE_AMBIGUITY_MARGIN,
        )

    def conflicts(self, embeddings, exclude_student_id):
        """Is this face already enrolled for a different active student?"""
        snapshot = self.snapshot(force=True)
        return conflicts_with_others(
            snapshot, embeddings, exclude_student_id, Config.FACE_DISTANCE_THRESHOLD
        )

    def stats(self):
        """Counts only. Safe for the health endpoint (no embeddings, no names)."""
        snap: Snapshot = self._snapshot
        return {
            "loaded": self._loaded,
            "students": snap.student_count,
            "samples": snap.sample_count,
            "legacy_samples": self._legacy_samples,
            "skipped_samples": snap.skipped,
            "built_at": self._built_at.isoformat() if self._built_at else None,
        }


face_index = FaceIndex()
