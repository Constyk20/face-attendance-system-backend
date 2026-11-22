"""Authentication routes.

POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
"""

from functools import wraps

from flask import Blueprint, jsonify, request

from database.db import get_db
from services.auth_service import (
    check_password,
    create_token,
    load_user_from_request,
    public_user,
)

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user, error = load_user_from_request()
        if error:
            return jsonify({"message": error}), 401
        return view(user, *args, **kwargs)

    return wrapped


def roles_required(*roles):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(user, *args, **kwargs):
            if user.get("role") not in roles:
                return jsonify({"message": "You do not have permission for this action"}), 403
            return view(user, *args, **kwargs)

        return wrapped

    return decorator


@auth_bp.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    # The hash is needed here, and only here.
    user = get_db().users.find_one({"email": email})
    if not user or not check_password(password, user.get("password_hash", "")):
        return jsonify({"message": "Invalid email or password"}), 401
    if not user.get("is_active", True):
        return jsonify({"message": "This account is deactivated"}), 403

    token = create_token(user)
    return jsonify({"token": token, "user": public_user(user)})


@auth_bp.post("/logout")
def logout():
    # JWT is stateless. The client must delete the token.
    return jsonify({"message": "Logged out"})


@auth_bp.get("/me")
@login_required
def me(user):
    return jsonify({"user": public_user(user)})
