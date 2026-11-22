"""In-memory runtime state for live attendance sessions.

Holds, per process:
  * a short-lived cache of validated sessions and lecturer->course access
  * a per-session "busy" flag and minimum interval (stops overlapping / spammy
    recognition requests)
  * the students already marked present in a session, so repeat sightings do
    not touch the database

This is single-process state. Run the backend as ONE process with several
threads (see serve.py). Authorization changes reach other processes only after
the TTL expires.
"""

import threading
import time

from config import Config

_MAX_RUNTIME_SESSIONS = 200
_MAX_MARKED_PER_SESSION = 5000


class _Runtime:
    __slots__ = ("busy", "last_started", "marked")

    def __init__(self):
        self.busy = False
        self.last_started = None
        self.marked = {}


class SessionState:
    def __init__(self, ttl=None, min_interval_ms=None, max_pending=None, clock=None):
        self._ttl = ttl
        self._min_interval_ms = min_interval_ms
        self._max_pending = max_pending
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._meta = {}  # session_id -> (data, stored_at)
        self._access = {}  # (user_id, course_id) -> (allowed, stored_at)
        self._runtime = {}  # session_id -> _Runtime
        self._inflight = 0

    # settings are read lazily so tests and .env changes are honoured
    @property
    def ttl(self):
        return Config.SESSION_CACHE_TTL_SECONDS if self._ttl is None else self._ttl

    @property
    def min_interval(self):
        ms = (
            Config.RECOGNITION_MIN_INTERVAL_MS
            if self._min_interval_ms is None
            else self._min_interval_ms
        )
        return ms / 1000.0

    @property
    def max_pending(self):
        return (
            Config.MAX_PENDING_RECOGNITIONS
            if self._max_pending is None
            else self._max_pending
        )

    # ----- validated session cache -----------------------------------
    def cached_session(self, session_id):
        now = self._clock()
        with self._lock:
            item = self._meta.get(session_id)
            if item is None:
                return None
            data, stored_at = item
            if now - stored_at > self.ttl:
                del self._meta[session_id]
                return None
            return data

    def store_session(self, session_id, data):
        with self._lock:
            self._meta[session_id] = (data, self._clock())

    def access(self, user_id, course_id):
        """Cached True/False, or None when unknown or expired."""
        now = self._clock()
        with self._lock:
            item = self._access.get((user_id, course_id))
            if item is None:
                return None
            allowed, stored_at = item
            if now - stored_at > self.ttl:
                del self._access[(user_id, course_id)]
                return None
            return allowed

    def store_access(self, user_id, course_id, allowed):
        with self._lock:
            self._access[(user_id, course_id)] = (bool(allowed), self._clock())

    # ----- overlap and rate control ----------------------------------
    def begin(self, session_id):
        """Try to start a recognition. Returns "ok", "busy" or "throttled"."""
        now = self._clock()
        with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is None:
                self._prune_locked()
                runtime = self._runtime[session_id] = _Runtime()
            if runtime.busy or self._inflight >= self.max_pending:
                return "busy"
            if (
                runtime.last_started is not None
                and now - runtime.last_started < self.min_interval
            ):
                return "throttled"
            runtime.busy = True
            runtime.last_started = now
            self._inflight += 1
            return "ok"

    def finish(self, session_id):
        with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is not None and runtime.busy:
                runtime.busy = False
                self._inflight = max(0, self._inflight - 1)

    # ----- students already marked -----------------------------------
    def marked(self, session_id, student_id):
        with self._lock:
            runtime = self._runtime.get(session_id)
            return None if runtime is None else runtime.marked.get(student_id)

    def mark(self, session_id, student_id, record):
        with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is None:
                runtime = self._runtime[session_id] = _Runtime()
            if len(runtime.marked) < _MAX_MARKED_PER_SESSION:
                runtime.marked[student_id] = record

    # ----- invalidation ----------------------------------------------
    def drop_session(self, session_id):
        with self._lock:
            self._meta.pop(session_id, None)
            runtime = self._runtime.pop(session_id, None)
            if runtime is not None and runtime.busy:
                self._inflight = max(0, self._inflight - 1)

    def drop_course(self, course_id):
        """Forget sessions and access rows for a course (it changed)."""
        with self._lock:
            for session_id, (data, _) in list(self._meta.items()):
                if str(data.get("course_id")) == course_id:
                    del self._meta[session_id]
            for key in [k for k in self._access if k[1] == course_id]:
                del self._access[key]

    def clear(self):
        with self._lock:
            self._meta.clear()
            self._access.clear()
            self._runtime.clear()
            self._inflight = 0

    def _prune_locked(self):
        if len(self._runtime) < _MAX_RUNTIME_SESSIONS:
            return
        idle = [
            (rt.last_started or 0, sid)
            for sid, rt in self._runtime.items()
            if not rt.busy
        ]
        for _, sid in sorted(idle)[: len(idle) // 2 or 1]:
            del self._runtime[sid]


session_state = SessionState()
