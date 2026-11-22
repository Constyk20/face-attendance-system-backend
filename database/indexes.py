"""All MongoDB indexes, created once at startup (never per request)."""

import logging

from pymongo.errors import AutoReconnect, PyMongoError, ServerSelectionTimeoutError

from database.db import get_db

log = logging.getLogger(__name__)


def _index_specs(db):
    """(collection, keys, options). Each one is justified by a real query."""
    return [
        # login by email; role filter + name sort in the lecturers list
        (db.users, [("email", 1)], {"unique": True}),
        (db.users, [("role", 1), ("full_name", 1)], {}),
        # student lookup, duplicate prevention, list sorted by name
        (db.students, [("matric_number", 1)], {"unique": True}),
        (
            db.students,
            [("email", 1)],
            {"unique": True, "partialFilterExpression": {"email": {"$type": "string"}}},
        ),
        (db.students, [("is_active", 1), ("full_name", 1)], {}),
        # embeddings are read by student at registration time
        (db.face_data, [("student_id", 1)], {}),
        # courses
        (db.courses, [("course_code", 1)], {"unique": True}),
        (db.courses, [("is_active", 1), ("course_code", 1)], {}),
        (db.course_assignments, [("course_id", 1), ("lecturer_id", 1)], {"unique": True}),
        (db.course_assignments, [("lecturer_id", 1)], {}),
        # sessions: lookups by course/status, and only ONE active session per course
        (db.attendance_sessions, [("course_id", 1), ("status", 1)], {}),
        (
            db.attendance_sessions,
            [("course_id", 1)],
            {
                "unique": True,
                "partialFilterExpression": {"status": "active"},
                "name": "one_active_session_per_course",
            },
        ),
        (db.attendance_sessions, [("started_at", -1)], {}),
        # attendance records: duplicate prevention + the report/dashboard queries
        (db.attendance_records, [("student_id", 1), ("session_id", 1)], {"unique": True}),
        (db.attendance_records, [("session_id", 1), ("created_at", 1)], {}),
        (db.attendance_records, [("course_id", 1), ("created_at", -1)], {}),
        (db.attendance_records, [("student_id", 1), ("created_at", -1)], {}),
        (db.attendance_records, [("date", 1)], {}),
        (db.attendance_records, [("created_at", -1)], {}),
    ]


def ensure_indexes():
    """Create every index. A failing index is logged and skipped, never fatal.

    create_index is idempotent, so running this at each start-up is cheap.
    The partial unique index fails if the database already contains two active
    sessions for one course; end the extra sessions and restart to enable it.
    """
    created, failed = 0, 0
    try:
        specs = _index_specs(get_db())
        for collection, keys, options in specs:
            try:
                collection.create_index(keys, **options)
                created += 1
            except (ServerSelectionTimeoutError, AutoReconnect):
                raise  # database is down: stop at once instead of waiting per index
            except PyMongoError as exc:
                failed += 1
                log.warning(
                    "Index %s on %s was not created: %s",
                    keys,
                    collection.name,
                    exc,
                )
    except PyMongoError:
        log.exception("Could not create indexes (is MongoDB running?)")
        return False
    log.info("Indexes ready (%d ok, %d skipped)", created, failed)
    return failed == 0
