"""Student records and facial registration.

GET    /api/students
POST   /api/students
GET    /api/students/<id>
PUT    /api/students/<id>
DELETE /api/students/<id>                 deactivates the student
POST   /api/students/<id>/register-face
"""

import logging
import re
from datetime import datetime, timezone

import cv2
import numpy as np
from bson import ObjectId
from flask import Blueprint, jsonify, request
from pymongo.errors import DuplicateKeyError

from config import Config
from database.db import get_db
from routes.auth import roles_required
from services.face_cache import face_index
from services.face_matching import consistency_outliers
from services.face_recognition import (
    MODEL_NAME,
    PIPELINE_VERSION,
    FaceError,
    decode_image,
    extract_embedding,
)
from utils import clean_text, pagination_meta, parse_object_id, parse_pagination

log = logging.getLogger(__name__)

students_bp = Blueprint("students", __name__, url_prefix="/api/students")

MATRIC_PATTERN = re.compile(r"^20[0-9]{2}/[0-9]{6}/reg$")
ALLOWED_LEVELS = {"100", "200", "300", "400", "500"}

# Fields kept in the face index: changing them must refresh the cache.
_CACHED_FIELDS = {"is_active", "full_name", "matric_number"}

_PUBLIC_FIELDS = {
    "full_name": 1,
    "matric_number": 1,
    "department": 1,
    "faculty": 1,
    "level": 1,
    "email": 1,
    "is_active": 1,
    "face_registered": 1,
    "created_at": 1,
    "updated_at": 1,
}


def public_student(student):
    return {
        "id": str(student["_id"]),
        "full_name": student.get("full_name", ""),
        "matric_number": student.get("matric_number", ""),
        "department": student.get("department", ""),
        "faculty": student.get("faculty", ""),
        "level": student.get("level", ""),
        "email": student.get("email"),
        "is_active": bool(student.get("is_active", True)),
        "face_registered": bool(student.get("face_registered", False)),
        "created_at": student.get("created_at"),
        "updated_at": student.get("updated_at"),
    }


def read_student_body(body, partial=False):
    """Return (fields, error_message)."""
    full_name = clean_text(body.get("full_name"))
    department = clean_text(body.get("department"))
    faculty = clean_text(body.get("faculty"))
    level = clean_text(body.get("level"), 3)
    raw_matric = clean_text(body.get("matric_number"), 20).lower()
    raw_email = clean_text(body.get("email"), 120).lower()

    required = {
        "full_name": full_name,
        "matric_number": raw_matric,
        "department": department,
        "faculty": faculty,
        "level": level,
    }
    if not partial:
        missing = [name for name, value in required.items() if not value]
        if missing:
            return None, "Missing fields: " + ", ".join(missing)

    fields = {}
    if full_name or not partial:
        if len(full_name) < 3:
            return None, "Full name must be at least 3 characters"
        fields["full_name"] = full_name
    if department or not partial:
        if not department:
            return None, "Department is required"
        fields["department"] = department
    if faculty or not partial:
        if not faculty:
            return None, "Faculty is required"
        fields["faculty"] = faculty
    if level or not partial:
        if level not in ALLOWED_LEVELS:
            return None, "Level must be 100, 200, 300, 400, or 500"
        fields["level"] = level
    if raw_matric or not partial:
        if not MATRIC_PATTERN.match(raw_matric):
            return None, "Matriculation number must look like 2021/129965/reg"
        fields["matric_number"] = raw_matric
    if "email" in body or not partial:
        if raw_email and "@" not in raw_email:
            return None, "Email address is not valid"
        fields["email"] = raw_email or None

    return fields, None


def build_search_query(search):
    """Search that can use the matric_number index when the text looks like one."""
    escaped = re.escape(search)
    if re.search(r"[0-9/]", search):
        # Matric numbers: anchored prefix, so MongoDB uses the unique index.
        return {"matric_number": {"$regex": "^" + re.escape(search.lower())}}
    return {
        "$or": [
            {"full_name": {"$regex": escaped, "$options": "i"}},
            {"department": {"$regex": escaped, "$options": "i"}},
        ]
    }


@students_bp.get("")
@roles_required("admin", "lecturer")
def list_students(user):
    query = {}
    status = request.args.get("status", "active")
    if status == "active":
        query["is_active"] = True
    elif status == "inactive":
        query["is_active"] = False

    search = clean_text(request.args.get("q"), 80)
    if search:
        query.update(build_search_query(search))

    page, limit, skip = parse_pagination(default_limit=200, max_limit=200)
    students_col = get_db().students
    total = students_col.count_documents(query)
    cursor = (
        students_col.find(query, _PUBLIC_FIELDS)
        .sort("full_name", 1)
        .skip(skip)
        .limit(limit)
    )
    return jsonify(
        {
            "students": [public_student(item) for item in cursor],
            **pagination_meta(page, limit, total),
        }
    )


@students_bp.post("")
@roles_required("admin")
def create_student(user):
    body = request.get_json(silent=True) or {}
    fields, error = read_student_body(body, partial=False)
    if error:
        return jsonify({"message": error}), 400

    now = datetime.now(timezone.utc)
    document = {
        **fields,
        "is_active": True,
        "face_registered": False,
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = get_db().students.insert_one(document)
    except DuplicateKeyError:
        return jsonify({"message": "Matriculation number or email already exists"}), 409

    document["_id"] = result.inserted_id
    return jsonify({"student": public_student(document)}), 201


@students_bp.get("/<student_id>")
@roles_required("admin", "lecturer")
def get_student(user, student_id):
    object_id = parse_object_id(student_id)
    if object_id is None:
        return jsonify({"message": "Invalid student id"}), 400
    student = get_db().students.find_one({"_id": object_id}, _PUBLIC_FIELDS)
    if not student:
        return jsonify({"message": "Student not found"}), 404
    return jsonify({"student": public_student(student)})


@students_bp.put("/<student_id>")
@roles_required("admin")
def update_student(user, student_id):
    object_id = parse_object_id(student_id)
    if object_id is None:
        return jsonify({"message": "Invalid student id"}), 400

    body = request.get_json(silent=True) or {}
    fields, error = read_student_body(body, partial=True)
    if error:
        return jsonify({"message": error}), 400
    if not fields and "is_active" not in body:
        return jsonify({"message": "No changes were sent"}), 400

    if "is_active" in body:
        fields["is_active"] = bool(body.get("is_active"))
    fields["updated_at"] = datetime.now(timezone.utc)

    try:
        result = get_db().students.update_one({"_id": object_id}, {"$set": fields})
    except DuplicateKeyError:
        return jsonify({"message": "Matriculation number or email already exists"}), 409
    if result.matched_count == 0:
        return jsonify({"message": "Student not found"}), 404

    if _CACHED_FIELDS & set(fields):
        face_index.invalidate()

    student = get_db().students.find_one({"_id": object_id}, _PUBLIC_FIELDS)
    return jsonify({"student": public_student(student)})


@students_bp.delete("/<student_id>")
@roles_required("admin")
def deactivate_student(user, student_id):
    object_id = parse_object_id(student_id)
    if object_id is None:
        return jsonify({"message": "Invalid student id"}), 400

    result = get_db().students.update_one(
        {"_id": object_id},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        return jsonify({"message": "Student not found"}), 404
    face_index.invalidate()  # an inactive student must stop matching immediately
    return jsonify({"message": "Student deactivated"})


@students_bp.post("/<student_id>/register-face")
@roles_required("admin")
def register_face(user, student_id):
    object_id = parse_object_id(student_id)
    if object_id is None:
        return jsonify({"message": "Invalid student id"}), 400

    db = get_db()
    student = db.students.find_one({"_id": object_id, "is_active": True}, {"_id": 1})
    if not student:
        return jsonify({"message": "Active student not found"}), 404

    body = request.get_json(silent=True) or {}
    images = body.get("images")
    if not isinstance(images, list) or len(images) < 3 or len(images) > 8:
        return jsonify({"message": "Send between 3 and 8 facial images"}), 400

    # 1. One embedding per image. Exactly one clear face is required in each.
    embeddings = []
    for index, image_value in enumerate(images, start=1):
        try:
            image = decode_image(image_value)
            embeddings.append(extract_embedding(image, policy="single"))
        except FaceError as exc:
            return jsonify({"message": f"Image {index}: {exc}"}), 400
        except (cv2.error, ValueError):
            log.exception("Registration image %d could not be processed", index)
            return jsonify({"message": f"Image {index} could not be processed"}), 400

    # 2. All images must show the same person.
    outliers = consistency_outliers(embeddings, Config.FACE_DISTANCE_THRESHOLD)
    if outliers:
        return jsonify(
            {
                "message": f"Image {outliers[0] + 1} does not look like the same person "
                "as the other images"
            }
        ), 400

    # 3. The face must not already belong to another active student.
    if face_index.conflicts(embeddings, object_id):
        return jsonify({"message": "This face is already registered to another student"}), 409

    # 4. Insert the new samples first, then remove the old ones, so a failure
    #    part-way never leaves the student without a face.
    now = datetime.now(timezone.utc)
    batch_id = ObjectId()
    db.face_data.insert_many(
        [
            {
                "student_id": object_id,
                "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                "model": MODEL_NAME,
                "pipeline_version": PIPELINE_VERSION,
                "batch_id": batch_id,
                "created_at": now,
            }
            for embedding in embeddings
        ]
    )
    db.face_data.delete_many({"student_id": object_id, "batch_id": {"$ne": batch_id}})
    db.students.update_one(
        {"_id": object_id},
        {"$set": {"face_registered": True, "updated_at": now}},
    )
    face_index.invalidate()
    log.info("Registered %d face samples for student %s", len(embeddings), object_id)
    return jsonify(
        {
            "message": "Face registered",
            "student_id": str(object_id),
            "samples": len(embeddings),
        }
    )
