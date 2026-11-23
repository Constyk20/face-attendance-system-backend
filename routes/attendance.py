"""Attendance sessions.

POST /api/attendance/start
POST /api/attendance/recognize
POST /api/attendance/end
GET  /api/attendance
GET  /api/attendance/<session_id>
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from pymongo.errors import DuplicateKeyError

from config import Config
from database.db import get_db
from routes.auth import roles_required
from services.attendance_service import identify_embedding
from services.face_recognition import FaceError
from services.session_state import session_state
from utils import pagination_meta, parse_object_id, parse_pagination

log = logging.getLogger(__name__)

attendance_bp = Blueprint("attendance", __name__, url_prefix="/api/attendance")

_COURSE_LABEL_FIELDS = {"course_code": 1, "course_title": 1}
_SESSION_FIELDS = {"course_id": 1, "lecturer_id": 1, "status": 1, "started_at": 1, "ended_at": 1}
_RECORD_FIELDS = {
    "student_id": 1,
    "student_name": 1,
    "matric_number": 1,
    "course_id": 1,
    "session_id": 1,
    "date": 1,
    "time": 1,
    "status": 1,
}


def lecturer_owns_course(user, course_id, use_cache=False):
    """Admins may use every course. Lecturers need an assignment.

    use_cache=True (used by the per-frame route only) accepts a result up to
    SESSION_CACHE_TTL_SECONDS old. Everything else asks the database.
    """
    if user.get("role") == "admin":
        return True
    user_key, course_key = str(user["_id"]), str(course_id)
    if use_cache:
        cached = session_state.access(user_key, course_key)
        if cached is not None:
            return cached
    allowed = (
        get_db().course_assignments.find_one(
            {"course_id": course_id, "lecturer_id": user["_id"]}, {"_id": 1}
        )
        is not None
    )
    session_state.store_access(user_key, course_key, allowed)
    return allowed


def visible_session_query(user):
    if user.get("role") == "admin":
        return {}
    rows = get_db().course_assignments.find({"lecturer_id": user["_id"]}, {"course_id": 1})
    return {"course_id": {"$in": [row["course_id"] for row in rows]}}


def public_session(session, course=None, lecturer_name=None):
    return {
        "id": str(session["_id"]),
        "course_id": str(session.get("course_id", "")),
        "course_code": None if not course else course.get("course_code"),
        "course_title": None if not course else course.get("course_title"),
        "lecturer_id": str(session.get("lecturer_id", "")),
        "lecturer_name": lecturer_name,
        "status": session.get("status"),
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
    }


def public_record(record):
    return {
        "id": str(record["_id"]),
        "student_id": str(record.get("student_id", "")),
        "student_name": record.get("student_name", ""),
        "matric_number": record.get("matric_number", ""),
        "course_id": str(record.get("course_id", "")),
        "session_id": str(record.get("session_id", "")),
        "date": record.get("date"),
        "time": record.get("time"),
        "status": record.get("status", "present"),
    }


def load_session_for_user(user, session_id):
    """Authoritative (uncached) load used by end / detail routes."""
    object_id = parse_object_id(session_id)
    if object_id is None:
        return None, (jsonify({"message": "Invalid session id"}), 400)
    session = get_db().attendance_sessions.find_one({"_id": object_id}, _SESSION_FIELDS)
    if not session:
        return None, (jsonify({"message": "Session not found"}), 404)
    if not lecturer_owns_course(user, session.get("course_id")):
        return None, (jsonify({"message": "You do not have permission for this course"}), 403)
    return session, None


def load_live_session(session_object_id):
    """Session facts needed per frame, cached briefly. Only ACTIVE sessions are cached."""
    key = str(session_object_id)
    data = session_state.cached_session(key)
    if data is not None:
        return data
    db = get_db()
    session = db.attendance_sessions.find_one({"_id": session_object_id}, _SESSION_FIELDS)
    if not session:
        return None
    course = db.courses.find_one({"_id": session.get("course_id")}, {"is_active": 1})
    data = {
        "id": session["_id"],
        "course_id": session.get("course_id"),
        "status": session.get("status"),
        "course_active": bool(course and course.get("is_active", True)),
    }
    if data["status"] == "active":
        session_state.store_session(key, data)
    return data


@attendance_bp.post("/start")
@roles_required("admin", "lecturer")
def start_session(user):
    body = request.get_json(silent=True) or {}
    course_id = parse_object_id(body.get("course_id"))
    if course_id is None:
        return jsonify({"message": "A valid course id is required"}), 400

    db = get_db()
    course = db.courses.find_one({"_id": course_id, "is_active": True}, _COURSE_LABEL_FIELDS)
    if not course:
        return jsonify({"message": "Active course not found"}), 404
    if not lecturer_owns_course(user, course_id):
        return jsonify({"message": "You are not assigned to this course"}), 403

    existing = db.attendance_sessions.find_one(
        {"course_id": course_id, "status": "active"}, _SESSION_FIELDS
    )
    if existing:
        return jsonify(
            {
                "message": "This course already has an active session",
                "session": public_session(existing, course),
            }
        ), 409

    now = datetime.now(timezone.utc)
    document = {
        "course_id": course_id,
        "lecturer_id": user["_id"],
        "status": "active",
        "started_at": now,
        "ended_at": None,
    }
    try:
        result = db.attendance_sessions.insert_one(document)
    except DuplicateKeyError:
        # Lost a race with another start request: the partial unique index caught it.
        existing = db.attendance_sessions.find_one(
            {"course_id": course_id, "status": "active"}, _SESSION_FIELDS
        )
        return jsonify(
            {
                "message": "This course already has an active session",
                "session": public_session(existing, course) if existing else None,
            }
        ), 409
    document["_id"] = result.inserted_id
    return jsonify({"session": public_session(document, course)}), 201


def _lightweight(**extra):
    """Small "not recognised" body. The frontend already understands recognized=false."""
    return jsonify({"recognized": False, **extra}), 200


def _recognized_response(student, record, duplicate, distance):
    payload = {
        "recognized": True,
        "duplicate": duplicate,
        "student": {
            "id": str(student["id"]),
            "full_name": student.get("full_name", ""),
            "matric_number": student.get("matric_number", ""),
        },
        "record": record,
    }
    if Config.RETURN_DISTANCE:
        payload["distance"] = distance
    return jsonify(payload)


@attendance_bp.post("/recognize")
@roles_required("admin", "lecturer")
def recognize(user):
    body = request.get_json(silent=True) or {}

    session_object_id = parse_object_id(body.get("session_id"))
    if session_object_id is None:
        return jsonify({"message": "Invalid session id"}), 400
    session = load_live_session(session_object_id)
    if not session:
        return jsonify({"message": "Session not found"}), 404
    if not lecturer_owns_course(user, session["course_id"], use_cache=True):
        return jsonify({"message": "You do not have permission for this course"}), 403
    if session["status"] != "active":
        return jsonify({"message": "This session is not active"}), 409
    if not session["course_active"]:
        return jsonify({"message": "This course is not active"}), 409

    embedding_value = body.get("embedding")
    if embedding_value is None:
        return jsonify({"message": "Face data is required"}), 400

    session_key = str(session_object_id)
    gate = session_state.begin(session_key)
    if gate == "busy":
        return _lightweight(busy=True, message="Still processing the previous frame")
    if gate == "throttled":
        return _lightweight(busy=True, throttled=True, message="Frames are arriving too fast")

    try:
        outcome = identify_embedding(embedding_value, body.get("model"))
    except FaceError as exc:  # malformed data or outdated browser model: safe to show
        return _lightweight(message=str(exc))
    finally:
        session_state.finish(session_key)

    if outcome.status != "match":
        return _lightweight(message=outcome.message)

    student = outcome.student
    student_key = str(student["id"])

    # Already marked in this session: answer from memory, no database work.
    remembered = session_state.marked(session_key, student_key)
    if remembered is not None:
        return _recognized_response(student, remembered, True, outcome.distance)

    now = datetime.now(timezone.utc)
    record = {
        "student_id": student["id"],
        "student_name": student.get("full_name", ""),
        "matric_number": student.get("matric_number", ""),
        "course_id": session["course_id"],
        "session_id": session_object_id,
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "status": "present",
        "distance": outcome.distance,
        "created_at": now,
    }
    records = get_db().attendance_records
    try:
        result = records.insert_one(record)
        record["_id"] = result.inserted_id
        duplicate = False
    except DuplicateKeyError:
        # The unique (student_id, session_id) index is the source of truth.
        record = records.find_one(
            {"student_id": student["id"], "session_id": session_object_id}, _RECORD_FIELDS
        )
        duplicate = True
        if record is None:  # deleted between the two calls: treat as unrecognised
            return _lightweight(message="Please try again")

    public = public_record(record)
    session_state.mark(session_key, student_key, public)
    return _recognized_response(student, public, duplicate, outcome.distance)


@attendance_bp.post("/end")
@roles_required("admin", "lecturer")
def end_session(user):
    body = request.get_json(silent=True) or {}
    session, error = load_session_for_user(user, body.get("session_id"))
    if error:
        return error

    now = datetime.now(timezone.utc)
    # Atomic: only one request can move the session from active to ended.
    result = get_db().attendance_sessions.update_one(
        {"_id": session["_id"], "status": "active"},
        {"$set": {"status": "ended", "ended_at": now}},
    )
    session_state.drop_session(str(session["_id"]))
    if result.matched_count == 0:
        return jsonify({"message": "This session is already closed"}), 409

    session["status"] = "ended"
    session["ended_at"] = now
    course = get_db().courses.find_one({"_id": session["course_id"]}, _COURSE_LABEL_FIELDS)
    return jsonify({"session": public_session(session, course)})


@attendance_bp.get("")
@roles_required("admin", "lecturer")
def list_sessions(user):
    query = visible_session_query(user)
    course_id = parse_object_id(request.args.get("course_id"))
    if request.args.get("course_id") and course_id is None:
        return jsonify({"message": "Invalid course id"}), 400
    if course_id is not None:
        if "course_id" in query and course_id not in query["course_id"]["$in"]:
            return jsonify({"sessions": [], **pagination_meta(1, 100, 0)})
        query["course_id"] = course_id

    db = get_db()
    page, limit, skip = parse_pagination(default_limit=100, max_limit=100)
    total = db.attendance_sessions.count_documents(query)
    sessions = list(
        db.attendance_sessions.find(query, _SESSION_FIELDS)
        .sort("started_at", -1)
        .skip(skip)
        .limit(limit)
    )
    courses = {
        item["_id"]: item
        for item in db.courses.find(
            {"_id": {"$in": list({row["course_id"] for row in sessions})}},
            _COURSE_LABEL_FIELDS,
        )
    }
    lecturers = {
        item["_id"]: item.get("full_name", "")
        for item in db.users.find(
            {"_id": {"$in": list({row["lecturer_id"] for row in sessions if row.get("lecturer_id")})}},
            {"full_name": 1},
        )
    }
    return jsonify(
        {
            "sessions": [
                public_session(
                    row,
                    courses.get(row.get("course_id")),
                    lecturers.get(row.get("lecturer_id")),
                )
                for row in sessions
            ],
            **pagination_meta(page, limit, total),
        }
    )


@attendance_bp.get("/<session_id>")
@roles_required("admin", "lecturer")
def get_session(user, session_id):
    session, error = load_session_for_user(user, session_id)
    if error:
        return error
    db = get_db()
    course = db.courses.find_one({"_id": session["course_id"]}, _COURSE_LABEL_FIELDS)
    records = (
        db.attendance_records.find({"session_id": session["_id"]}, _RECORD_FIELDS)
        .sort("created_at", 1)
        .limit(5000)
    )
    return jsonify(
        {
            "session": public_session(session, course),
            "records": [public_record(row) for row in records],
        }
    )
