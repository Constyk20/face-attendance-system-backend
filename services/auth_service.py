"""Password hashing and JWT helpers. No HTTP code lives here."""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from bson import ObjectId
from bson.errors import InvalidId
from flask import request

from config import Config
from database.db import get_db


def hash_password(plain_password):
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain_password, password_hash):
    if not plain_password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except ValueError:  # malformed hash in the database
        return False


def create_token(user):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["_id"]),
        "role": user["role"],
        "email": user["email"],
        "iat": now,
        "exp": now + timedelta(hours=Config.JWT_EXPIRES_HOURS),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm="HS256")


def decode_token(token):
    return jwt.decode(
        token,
        Config.JWT_SECRET,
        algorithms=["HS256"],
        options={"require": ["exp", "sub"]},
    )


def public_user(user):
    return {
        "id": str(user["_id"]),
        "full_name": user.get("full_name", ""),
        "email": user.get("email", ""),
        "role": user.get("role", ""),
        "is_active": bool(user.get("is_active", True)),
    }


def get_bearer_token():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header.split(" ", 1)[1].strip()
    return token or None


def load_user_from_request():
    """Return (user_dict, error_message). error_message is None on success.

    The user is read on every request (so a deactivated account is locked out
    immediately), but without the password hash. A database outage raises
    PyMongoError and becomes a 503, not a misleading 401.
    """
    token = get_bearer_token()
    if not token:
        return None, "Missing bearer token"
    try:
        payload = decode_token(token)
        user_id = ObjectId(payload["sub"])
    except (jwt.PyJWTError, KeyError, InvalidId, TypeError):
        return None, "Invalid or expired token"

    user = get_db().users.find_one(
        {"_id": user_id}, {"password_hash": 0}
    )
    if not user or not user.get("is_active", True):
        return None, "User is not active"
    return user, None
