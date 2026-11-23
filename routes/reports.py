"""Attendance reports and dashboard stats.

GET /api/reports/attendance
GET /api/reports/student/<student_id>
GET /api/reports/course/<course_id>
GET /api/reports/export
GET /api/reports/dashboard
"""

import csv
import io
import logging
from datetime import datetime, timedelta, timezone

from flask import Blueprint, Response, jsonify, request

from config import Config
from database.db import get_db
from routes.auth import roles_required
from utils import (
    pagination_meta,
    parse_date_filters,
    parse_object_id,
    parse_pagination,
)

log = logging.getLogger(__name__)

reports_bp = Blueprint("reports", __name__, url_prefix="/api/reports")

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


def lecturer_course_ids(user):
    """None for admins (no restriction), otherwise the list of assigned course ids."""
    if user.get("role") == "admin":
        return None
    rows = get_db().course_assignments.find({"lecturer_id": user["_id"]}, {"course_id": 1})
    return [row["course_id"] for row in rows]


def _record_query(user, allowed):
    """Base attendance_records query restricted to the user's courses."""
    return {} if allowed is None else {"course_id": {"$in": allowed}}


def _course_filter(allowed):
    """Return (course_id or None, error_response or None) from ?course_id=."""
    raw = request.args.get("course_id")
    if not raw:
        return None, None
    course_id = parse_object_id(raw)
    if course_id is None:
        return None, (jsonify({"message": "Invalid course id"}), 400)
    if allowed is not None and course_id not in allowed:
        return None, (jsonify({"message": "You do not have access to this course"}), 403)
    return course_id, None


def _record_json(row):
    return {
        "id": str(row["_id"]),
        "student_id": str(row.get("student_id", "")),
        "student_name": row.get("student_name", ""),
        "matric_number": row.get("matric_number", ""),
        "course_id": str(row.get("course_id", "")),
        "session_id": str(row.get("session_id", "")),
        "date": row.get("date"),
        "time": row.get("time"),
        "status": row.get("status", "present"),
    }


@reports_bp.get("/dashboard")
@roles_required("admin", "lecturer")
def dashboard(user):
    db = get_db()
    allowed = lecturer_course_ids(user)
    is_admin = allowed is None
    course_match = {} if is_admin else {"course_id": {"$in": allowed}}
    today_date = datetime.now(timezone.utc).date()
    day_strings = [
        (today_date - timedelta(days=offset)).isoformat() for offset in range(6, -1, -1)
    ]

    # Present marks per day for the last 7 days (also gives "present today").
    weekly_counts = {
        row["_id"]: row["count"]
        for row in db.attendance_records.aggregate(
            [
                {"$match": {"date": {"$in": day_strings}, **course_match}},
                {"$group": {"_id": "$date", "count": {"$sum": 1}}},
            ]
        )
    }
    weekly = [{"date": day, "present": weekly_counts.get(day, 0)} for day in day_strings]
    present_today = weekly[-1]["present"]

    # Total and active sessions in ONE aggregation instead of two counts.
    session_stats = list(
        db.attendance_sessions.aggregate(
            [
                {"$match": course_match},
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "active": {"$sum": {"$cond": [{"$eq": ["$status", "active"]}, 1, 0]}},
                    }
                },
            ]
        )
    )
    total_sessions = session_stats[0]["total"] if session_stats else 0
    active_sessions = session_stats[0]["active"] if session_stats else 0

    recent = list(
        db.attendance_records.find(course_match, _RECORD_FIELDS).sort("created_at", -1).limit(10)
    )
    total_students = db.students.count_documents({"is_active": True})
    if is_admin:
        total_lecturers = db.users.count_documents({"role": "lecturer", "is_active": True})
        total_courses = db.courses.count_documents({"is_active": True})
    else:
        total_lecturers = 0
        total_courses = len(allowed)

    return jsonify(
        {
            "stats": {
                "total_students": total_students,
                "total_lecturers": total_lecturers,
                "total_courses": total_courses,
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "present_today": present_today,
            },
            "weekly": weekly,
            "recent_activity": [
                {
                    "id": str(row["_id"]),
                    "student_name": row.get("student_name", ""),
                    "matric_number": row.get("matric_number", ""),
                    "date": row.get("date"),
                    "time": row.get("time"),
                    "status": row.get("status", "present"),
                    "course_id": str(row.get("course_id", "")),
                    "session_id": str(row.get("session_id", "")),
                }
                for row in recent
            ],
        }
    )


@reports_bp.get("/attendance")
@roles_required("admin", "lecturer")
def attendance_report(user):
    allowed = lecturer_course_ids(user)
    query = _record_query(user, allowed)

    course_id, error = _course_filter(allowed)
    if error:
        return error
    if course_id is not None:
        query["course_id"] = course_id

    raw_session = request.args.get("session_id")
    if raw_session:
        session_id = parse_object_id(raw_session)
        if session_id is None:
            return jsonify({"message": "Invalid session id"}), 400
        query["session_id"] = session_id

    date_filters, date_error = parse_date_filters(request.args)
    if date_error:
        return jsonify({"message": date_error}), 400
    if date_filters:
        query["date"] = date_filters

    page, limit, skip = parse_pagination(default_limit=500, max_limit=500)
    records_col = get_db().attendance_records
    total = records_col.count_documents(query)
    records = list(
        records_col.find(query, _RECORD_FIELDS).sort("created_at", -1).skip(skip).limit(limit)
    )
    return jsonify(
        {
            "count": len(records),
            "records": [_record_json(row) for row in records],
            **pagination_meta(page, limit, total),
        }
    )


@reports_bp.get("/student/<student_id>")
@roles_required("admin", "lecturer")
def student_report(user, student_id):
    object_id = parse_object_id(student_id)
    if object_id is None:
        return jsonify({"message": "Invalid student id"}), 400

    db = get_db()
    student = db.students.find_one(
        {"_id": object_id},
        {"full_name": 1, "matric_number": 1, "department": 1, "level": 1},
    )
    if not student:
        return jsonify({"message": "Student not found"}), 404

    allowed = lecturer_course_ids(user)
    query = {"student_id": object_id, **_record_query(user, allowed)}

    # Exact per-course totals over ALL records (the list below is the latest 200).
    totals = list(
        db.attendance_records.aggregate(
            [
                {"$match": query},
                {"$group": {"_id": "$course_id", "present_count": {"$sum": 1}}},
            ]
        )
    )
    records = list(
        db.attendance_records.find(
            query, {"course_id": 1, "session_id": 1, "date": 1, "time": 1, "status": 1}
        )
        .sort("created_at", -1)
        .limit(200)
    )

    session_ids = list({row["session_id"] for row in records if row.get("session_id")})
    sessions = {
        row["_id"]: row
        for row in db.attendance_sessions.find({"_id": {"$in": session_ids}}, {"status": 1})
    }
    course_ids = list({row["_id"] for row in totals if row["_id"]})
    courses = {
        row["_id"]: row
        for row in db.courses.find(
            {"_id": {"$in": course_ids}}, {"course_code": 1, "course_title": 1}
        )
    }

    by_course = []
    for row in totals:
        course = courses.get(row["_id"], {})
        by_course.append(
            {
                "course_id": str(row["_id"]) if row["_id"] else "",
                "course_code": course.get("course_code", ""),
                "course_title": course.get("course_title", ""),
                "present_count": row["present_count"],
            }
        )

    return jsonify(
        {
            "student": {
                "id": str(student["_id"]),
                "full_name": student.get("full_name", ""),
                "matric_number": student.get("matric_number", ""),
                "department": student.get("department", ""),
                "level": student.get("level", ""),
            },
            "total_present": sum(item["present_count"] for item in by_course),
            "by_course": by_course,
            "records": [
                {
                    "id": str(row["_id"]),
                    "course_code": courses.get(row.get("course_id"), {}).get("course_code", ""),
                    "date": row.get("date"),
                    "time": row.get("time"),
                    "status": row.get("status", "present"),
                    "session_status": sessions.get(row.get("session_id"), {}).get("status"),
                }
                for row in records
            ],
        }
    )


@reports_bp.get("/course/<course_id>")
@roles_required("admin", "lecturer")
def course_report(user, course_id):
    object_id = parse_object_id(course_id)
    if object_id is None:
        return jsonify({"message": "Invalid course id"}), 400

    allowed = lecturer_course_ids(user)
    if allowed is not None and object_id not in allowed:
        return jsonify({"message": "You do not have access to this course"}), 403

    db = get_db()
    course = db.courses.find_one(
        {"_id": object_id},
        {"course_code": 1, "course_title": 1, "department": 1, "level": 1},
    )
    if not course:
        return jsonify({"message": "Course not found"}), 404

    session_count = db.attendance_sessions.count_documents({"course_id": object_id})
    sessions = list(
        db.attendance_sessions.find(
            {"course_id": object_id}, {"status": 1, "started_at": 1, "ended_at": 1}
        )
        .sort("started_at", -1)
        .limit(500)
    )

    # Grouping happens inside MongoDB; no record is loaded into Python.
    grouped = db.attendance_records.aggregate(
        [
            {"$match": {"course_id": object_id}},
            {
                "$group": {
                    "_id": "$student_id",
                    "student_name": {"$first": "$student_name"},
                    "matric_number": {"$first": "$matric_number"},
                    "present_count": {"$sum": 1},
                }
            },
            {"$sort": {"student_name": 1}},
        ]
    )
    students = []
    total_marks = 0
    for item in grouped:
        total_marks += item["present_count"]
        students.append(
            {
                "student_id": str(item["_id"]),
                "student_name": item.get("student_name", ""),
                "matric_number": item.get("matric_number", ""),
                "present_count": item["present_count"],
                "session_count": session_count,
                "percentage": round((item["present_count"] / session_count) * 100, 1)
                if session_count
                else 0,
            }
        )

    return jsonify(
        {
            "course": {
                "id": str(course["_id"]),
                "course_code": course.get("course_code", ""),
                "course_title": course.get("course_title", ""),
                "department": course.get("department", ""),
                "level": course.get("level", ""),
            },
            "session_count": session_count,
            "total_present_marks": total_marks,
            "students": students,
            "sessions": [
                {
                    "id": str(row["_id"]),
                    "status": row.get("status"),
                    "started_at": row.get("started_at"),
                    "ended_at": row.get("ended_at"),
                }
                for row in sessions
            ],
        }
    )


_CSV_HEADER = [
    "Student Name",
    "Matric Number",
    "Course Code",
    "Course Title",
    "Date",
    "Time",
    "Status",
]


def _csv_cell(value):
    """Stop spreadsheet formula injection (=, +, -, @ at the start of a cell)."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


@reports_bp.get("/export")
@roles_required("admin", "lecturer")
def export_csv(user):
    db = get_db()
    allowed = lecturer_course_ids(user)
    query = _record_query(user, allowed)

    course_id, error = _course_filter(allowed)
    if error:
        return error
    if course_id is not None:
        query["course_id"] = course_id

    date_filters, date_error = parse_date_filters(request.args)
    if date_error:
        return jsonify({"message": date_error}), 400
    if date_filters:
        query["date"] = date_filters

    course_query = {"_id": course_id} if course_id is not None else (
        {} if allowed is None else {"_id": {"$in": allowed}}
    )
    courses = {
        row["_id"]: row
        for row in db.courses.find(course_query, {"course_code": 1, "course_title": 1})
    }
    cursor = (
        db.attendance_records.find(
            query,
            {
                "student_name": 1,
                "matric_number": 1,
                "course_id": 1,
                "date": 1,
                "time": 1,
                "status": 1,
            },
        )
        .sort("created_at", -1)
        .limit(Config.EXPORT_MAX_ROWS)
        .batch_size(1000)
    )

    def generate():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(_CSV_HEADER)
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        try:
            for index, row in enumerate(cursor, start=1):
                course = courses.get(row.get("course_id"), {})
                writer.writerow(
                    [
                        _csv_cell(row.get("student_name", "")),
                        _csv_cell(row.get("matric_number", "")),
                        _csv_cell(course.get("course_code", "")),
                        _csv_cell(course.get("course_title", "")),
                        row.get("date", ""),
                        row.get("time", ""),
                        row.get("status", "present"),
                    ]
                )
                if index % 500 == 0:
                    yield buffer.getvalue()
                    buffer.seek(0)
                    buffer.truncate(0)
        except Exception:
            # Headers are already sent, so all we can do is log and stop the file.
            log.exception("CSV export stopped part-way")
            return
        finally:
            cursor.close()
        tail = buffer.getvalue()
        if tail:
            yield tail

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_report.csv"},
    )
