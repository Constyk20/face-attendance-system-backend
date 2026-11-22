"""Courses and lecturer assignment.

GET    /api/courses
POST   /api/courses
PUT    /api/courses/<id>
DELETE /api/courses/<id>     deactivates the course
"""

import re
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from pymongo.errors import DuplicateKeyError

from database.db import get_db
from routes.auth import roles_required
from services.session_state import session_state
from utils import clean_text, parse_object_id

courses_bp = Blueprint("courses", __name__, url_prefix="/api/courses")

ALLOWED_LEVELS = {"100", "200", "300", "400", "500"}
CODE_PATTERN = re.compile(r"^[A-Za-z]{2,6}\s?[0-9]{3}$")

_COURSE_FIELDS = {
    "course_code": 1,
    "course_title": 1,
    "department": 1,
    "level": 1,
    "is_active": 1,
}
_LECTURER_FIELDS = {"full_name": 1, "email": 1}


def public_course(course, lecturer=None):
    return {
        "id": str(course["_id"]),
        "course_code": course.get("course_code", ""),
        "course_title": course.get("course_title", ""),
        "department": course.get("department", ""),
        "level": course.get("level", ""),
        "is_active": bool(course.get("is_active", True)),
        "lecturer": None
        if not lecturer
        else {
            "id": str(lecturer["_id"]),
            "full_name": lecturer.get("full_name", ""),
            "email": lecturer.get("email", ""),
        },
    }


def lecturer_map(lecturer_ids):
    if not lecturer_ids:
        return {}
    found = get_db().users.find(
        {"_id": {"$in": list(lecturer_ids)}, "role": "lecturer"}, _LECTURER_FIELDS
    )
    return {item["_id"]: item for item in found}


def assigned_lecturer_id(course_id):
    row = get_db().course_assignments.find_one({"course_id": course_id}, {"lecturer_id": 1})
    return None if not row else row.get("lecturer_id")


def save_assignment(course_id, lecturer_id):
    """Make `lecturer_id` the only assigned lecturer, without a moment of "no lecturer"."""
    db = get_db()
    db.course_assignments.delete_many(
        {"course_id": course_id, "lecturer_id": {"$ne": lecturer_id}}
    )
    db.course_assignments.update_one(
        {"course_id": course_id, "lecturer_id": lecturer_id},
        {"$setOnInsert": {"assigned_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    session_state.drop_course(str(course_id))


def read_course_body(body, partial=False):
    code = clean_text(body.get("course_code"), 12).upper().replace(" ", "")
    title = clean_text(body.get("course_title"))
    department = clean_text(body.get("department"))
    level = clean_text(body.get("level"), 3)
    lecturer_raw = clean_text(body.get("lecturer_id"), 32)

    if not partial:
        missing = [
            name
            for name, value in {
                "course_code": code,
                "course_title": title,
                "department": department,
                "level": level,
                "lecturer_id": lecturer_raw,
            }.items()
            if not value
        ]
        if missing:
            return None, None, "Missing fields: " + ", ".join(missing)

    fields = {}
    lecturer_id = None
    if code or not partial:
        if not CODE_PATTERN.match(code):
            return None, None, "Course code must look like CSC401"
        fields["course_code"] = code
    if title or not partial:
        if len(title) < 3:
            return None, None, "Course title must be at least 3 characters"
        fields["course_title"] = title
    if department or not partial:
        if not department:
            return None, None, "Department is required"
        fields["department"] = department
    if level or not partial:
        if level not in ALLOWED_LEVELS:
            return None, None, "Level must be 100, 200, 300, 400, or 500"
        fields["level"] = level
    if lecturer_raw or not partial:
        lecturer_id = parse_object_id(lecturer_raw)
        if lecturer_id is None:
            return None, None, "Invalid lecturer id"
        lecturer = get_db().users.find_one(
            {"_id": lecturer_id, "role": "lecturer", "is_active": True}, {"_id": 1}
        )
        if not lecturer:
            return None, None, "Active lecturer not found"

    return fields, lecturer_id, None


@courses_bp.get("")
@roles_required("admin", "lecturer")
def list_courses(user):
    db = get_db()
    query = {}
    if request.args.get("status", "active") == "active":
        query["is_active"] = True
    elif request.args.get("status") == "inactive":
        query["is_active"] = False

    if user.get("role") == "lecturer":
        assigned = db.course_assignments.find({"lecturer_id": user["_id"]}, {"course_id": 1})
        query["_id"] = {"$in": [row["course_id"] for row in assigned]}

    courses = list(db.courses.find(query, _COURSE_FIELDS).sort("course_code", 1).limit(200))
    assignment_rows = db.course_assignments.find(
        {"course_id": {"$in": [course["_id"] for course in courses]}},
        {"course_id": 1, "lecturer_id": 1},
    )
    by_course = {row["course_id"]: row["lecturer_id"] for row in assignment_rows}
    people = lecturer_map(set(by_course.values()))

    return jsonify(
        {
            "courses": [
                public_course(course, people.get(by_course.get(course["_id"])))
                for course in courses
            ]
        }
    )


@courses_bp.post("")
@roles_required("admin")
def create_course(user):
    body = request.get_json(silent=True) or {}
    fields, lecturer_id, error = read_course_body(body, partial=False)
    if error:
        return jsonify({"message": error}), 400

    now = datetime.now(timezone.utc)
    document = {**fields, "is_active": True, "created_at": now, "updated_at": now}
    try:
        result = get_db().courses.insert_one(document)
    except DuplicateKeyError:
        return jsonify({"message": "Course code already exists"}), 409

    document["_id"] = result.inserted_id
    save_assignment(document["_id"], lecturer_id)
    lecturer = get_db().users.find_one({"_id": lecturer_id}, _LECTURER_FIELDS)
    return jsonify({"course": public_course(document, lecturer)}), 201


@courses_bp.put("/<course_id>")
@roles_required("admin")
def update_course(user, course_id):
    object_id = parse_object_id(course_id)
    if object_id is None:
        return jsonify({"message": "Invalid course id"}), 400

    body = request.get_json(silent=True) or {}
    fields, lecturer_id, error = read_course_body(body, partial=True)
    if error:
        return jsonify({"message": error}), 400
    if not fields and lecturer_id is None and "is_active" not in body:
        return jsonify({"message": "No changes were sent"}), 400

    if "is_active" in body:
        fields["is_active"] = bool(body.get("is_active"))
    fields["updated_at"] = datetime.now(timezone.utc)

    try:
        result = get_db().courses.update_one({"_id": object_id}, {"$set": fields})
    except DuplicateKeyError:
        return jsonify({"message": "Course code already exists"}), 409
    if result.matched_count == 0:
        return jsonify({"message": "Course not found"}), 404

    if lecturer_id is not None:
        save_assignment(object_id, lecturer_id)
    session_state.drop_course(str(object_id))  # active flag may have changed

    course = get_db().courses.find_one({"_id": object_id}, _COURSE_FIELDS)
    current_lecturer_id = assigned_lecturer_id(object_id)
    lecturer = None
    if current_lecturer_id is not None:
        lecturer = get_db().users.find_one({"_id": current_lecturer_id}, _LECTURER_FIELDS)
    return jsonify({"course": public_course(course, lecturer)})


@courses_bp.delete("/<course_id>")
@roles_required("admin")
def deactivate_course(user, course_id):
    object_id = parse_object_id(course_id)
    if object_id is None:
        return jsonify({"message": "Invalid course id"}), 400
    result = get_db().courses.update_one(
        {"_id": object_id},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        return jsonify({"message": "Course not found"}), 404
    session_state.drop_course(str(object_id))
    return jsonify({"message": "Course deactivated"})
