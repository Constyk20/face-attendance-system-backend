"""Lecturer accounts. Admin only.

GET  /api/lecturers
POST /api/lecturers
"""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from pymongo.errors import DuplicateKeyError

from database.db import get_db
from routes.auth import roles_required
from services.auth_service import hash_password, public_user

lecturers_bp = Blueprint("lecturers", __name__, url_prefix="/api/lecturers")

_LIST_FIELDS = {"full_name": 1, "email": 1, "role": 1, "is_active": 1}


@lecturers_bp.get("")
@roles_required("admin")
def list_lecturers(user):
    lecturers = (
        get_db()
        .users.find({"role": "lecturer"}, _LIST_FIELDS)
        .sort("full_name", 1)
        .limit(1000)
    )
    return jsonify({"lecturers": [public_user(item) for item in lecturers]})


@lecturers_bp.post("")
@roles_required("admin")
def create_lecturer(user):
    body = request.get_json(silent=True) or {}
    full_name = str(body.get("full_name", "")).strip()
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))

    if len(full_name) < 3:
        return jsonify({"message": "Full name must be at least 3 characters"}), 400
    if "@" not in email:
        return jsonify({"message": "A valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters"}), 400
    if len(password.encode("utf-8")) > 72:
        # bcrypt silently ignores everything after 72 bytes
        return jsonify({"message": "Password must be 72 bytes or fewer"}), 400

    now = datetime.now(timezone.utc)
    document = {
        "full_name": full_name,
        "email": email,
        "password_hash": hash_password(password),
        "role": "lecturer",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = get_db().users.insert_one(document)
    except DuplicateKeyError:
        return jsonify({"message": "Email already exists"}), 409

    document["_id"] = result.inserted_id
    return jsonify({"lecturer": public_user(document)}), 201
