import logging
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Keep TensorFlow's start-up chatter out of the console (must be set before import).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo.errors import PyMongoError
from werkzeug.exceptions import HTTPException

from config import Config
from database.db import ping_database
from database.indexes import ensure_indexes
from routes.attendance import attendance_bp
from routes.auth import auth_bp
from routes.courses import courses_bp
from routes.lecturers import lecturers_bp
from routes.reports import reports_bp
from routes.students import students_bp
from services import face_recognition
from services.face_cache import face_index
from utils import setup_logging

log = logging.getLogger("app")

_HTTP_MESSAGES = {
    400: "Bad request",
    404: "Resource not found",
    405: "Method not allowed",
    413: "Request is too large",
    415: "Content-Type must be application/json",
    429: "Too many requests",
}


def create_app():
    setup_logging()
    Config.validate()

    app = Flask(__name__)
    app.config.from_object(Config)  # includes MAX_CONTENT_LENGTH (-> 413)

    CORS(
        app,
        resources={r"/api/*": {"origins": Config.FRONTEND_ORIGINS}},
        supports_credentials=True,
    )

    @app.before_request
    def require_json_body():
        if request.method in ("POST", "PUT", "PATCH") and request.content_length:
            if not request.is_json:
                return jsonify({"message": _HTTP_MESSAGES[415]}), 415

    app.register_blueprint(auth_bp)
    app.register_blueprint(students_bp)
    app.register_blueprint(lecturers_bp)
    app.register_blueprint(courses_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(reports_bp)

    # ----- JSON errors (never HTML, never internal details) ------------
    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        code = error.code or 500
        return jsonify({"message": _HTTP_MESSAGES.get(code, error.name)}), code

    @app.errorhandler(PyMongoError)
    def handle_database_error(error):
        log.error("Database error: %s", error.__class__.__name__, exc_info=True)
        return jsonify({"message": "Database temporarily unavailable"}), 503

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        log.exception("Unhandled error")
        return jsonify({"message": "Internal server error"}), 500

    # ----- Start-up work, done ONCE per serving process ----------------
    # With the Werkzeug reloader the parent process only supervises; skip the
    # heavy work there so indexes and the model are not handled twice.
    reloader_parent = (
        Config.DEBUG
        and Config.USE_RELOADER
        and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    )
    if not reloader_parent:
        ensure_indexes()  # no model to load: face inference runs in the browser

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "face-attendance-api",
                "stage": 7,
                "model": face_recognition.model_status(),
                "embedding_model": face_recognition.EMBEDDING_MODEL,
                "face_cache": face_index.stats(),
            }
        )

    @app.get("/api/health/db")
    def health_db():
        ok, message = ping_database()
        body = {
            "status": "ok" if ok else "error",
            "database": Config.MONGODB_DB_NAME,
            "message": message,
        }
        return jsonify(body), (200 if ok else 503)

    return app


app = create_app()


if __name__ == "__main__":
    # Development server. For real use run:  python serve.py   (waitress)
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=Config.USE_RELOADER,
        threaded=True,
    )
