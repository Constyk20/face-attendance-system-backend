"""Small helpers shared by routes. No HTTP routes live here."""

import logging
import logging.handlers
import re

from bson import ObjectId
from bson.errors import InvalidId
from flask import request

from config import Config

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_object_id(value):
    """Return an ObjectId, or None for anything that is not a valid id string."""
    if not isinstance(value, (str, ObjectId)):
        return None
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def clean_text(value, max_length=120):
    return str(value or "").strip()[:max_length]


def parse_pagination(default_limit=50, max_limit=200):
    """Read ?page= and ?limit= safely. Returns (page, limit, skip)."""

    def read(name, default):
        try:
            return int(request.args.get(name, default))
        except (TypeError, ValueError):
            return default

    page = max(1, read("page", 1))
    limit = min(max(1, read("limit", default_limit)), max_limit)
    return page, limit, (page - 1) * limit


def pagination_meta(page, limit, total):
    pages = max(1, -(-total // limit)) if limit else 1
    return {"page": page, "limit": limit, "total": total, "pages": pages}


def parse_date_filters(args):
    """Return (filters, error_message) for ?start_date= / ?end_date= (YYYY-MM-DD)."""
    filters = {}
    for key, operator in (("start_date", "$gte"), ("end_date", "$lte")):
        value = args.get(key)
        if not value:
            continue
        if not DATE_PATTERN.match(value):
            return None, f"{key} must look like 2026-01-31"
        filters[operator] = value
    return filters, None


def setup_logging():
    """Configure root logging once. Never log tokens, passwords or embeddings."""
    root = logging.getLogger()
    if getattr(root, "_face_attendance_configured", False):
        return
    level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if Config.LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            Config.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet very chatty libraries.
    for noisy in ("pymongo", "urllib3", "h5py", "tensorflow"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    root._face_attendance_configured = True
