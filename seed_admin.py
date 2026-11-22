"""Create the first administrator from ADMIN_* values in .env."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

from database.db import get_db
from services.auth_service import hash_password

load_dotenv()


def ensure_indexes(db):
    db.users.create_index("email", unique=True)


def main():
    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "")
    full_name = os.getenv("ADMIN_NAME", "System Administrator").strip()

    if not email or not password:
        raise SystemExit("Set ADMIN_EMAIL and ADMIN_PASSWORD in .env first.")

    db = get_db()
    ensure_indexes(db)

    existing = db.users.find_one({"email": email})
    if existing:
        print(f"Admin already exists: {email}")
        return

    now = datetime.now(timezone.utc)
    try:
        db.users.insert_one(
            {
                "full_name": full_name,
                "email": email,
                "password_hash": hash_password(password),
                "role": "admin",
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
        )
    except DuplicateKeyError:
        print(f"Admin already exists: {email}")
        return

    print(f"Admin created: {email}")


if __name__ == "__main__":
    main()