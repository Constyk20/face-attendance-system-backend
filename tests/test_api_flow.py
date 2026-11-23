"""End-to-end API flow with an in-memory MongoDB (mongomock): no model, no camera."""
import os

os.environ.setdefault("FLASK_DEBUG", "1")

import mongomock
import numpy as np
import pytest
from bson import ObjectId

import database.db as dbmod
from config import Config

MODEL = Config.EMBEDDING_MODEL


def unit(rng, base=None, noise=0.0):
    v = rng.normal(size=128) if base is None else np.array(base) + rng.normal(size=128) * noise
    return (v / np.linalg.norm(v)).round(6).tolist()


@pytest.fixture()
def env(monkeypatch):
    client = mongomock.MongoClient()
    monkeypatch.setattr(dbmod, "get_client", lambda: client)
    monkeypatch.setattr("database.indexes.ensure_indexes", lambda: None, raising=False)
    import sys

    for name in [m for m in sys.modules if m == "app" or m.startswith("services.face_cache")]:
        sys.modules.pop(name)
    import app as appmod

    from services.session_state import session_state
    return appmod.app.test_client(), client[Config.MONGODB_DB_NAME]


def test_health_has_no_model_loading(env):
    client, _ = env
    body = client.get("/api/health").get_json()
    assert body["model"] == "browser" and body["embedding_model"] == MODEL


def _admin(db, client):
    from services.auth_service import create_token, hash_password
    from datetime import datetime, timezone

    doc = {"email": "a@x.com", "full_name": "Admin", "role": "admin", "is_active": True,
           "password_hash": hash_password("pw123456"), "created_at": datetime.now(timezone.utc)}
    doc["_id"] = db.users.insert_one(doc).inserted_id
    return {"Authorization": "Bearer " + create_token(doc)}


def _student(client, h, name, matric):
    r = client.post("/api/students", headers=h, json={
        "full_name": name, "matric_number": matric, "department": "CS", "faculty": "Sci", "level": "300"})
    assert r.status_code == 201, r.get_json()
    return r.get_json()["student"]["id"]


def test_full_flow_and_legacy_transition(env):
    client, db = env
    h = _admin(db, client)
    rng = np.random.default_rng(5)
    alice_base, bob_base = unit(rng), unit(rng)
    alice = _student(client, h, "Alice Ade", "2021/111111/reg")
    bob = _student(client, h, "Bob Bello", "2021/222222/reg")
    old = _student(client, h, "Old Timer", "2021/333333/reg")

    # a legacy (Facenet) student: face_registered but different model
    db.students.update_one({"_id": ObjectId(old)}, {"$set": {"face_registered": True}})
    db.face_data.insert_one({"student_id": ObjectId(old), "embedding": unit(rng), "model": "Facenet"})

    def reg(sid, base, model=MODEL, n=3):
        return client.post(f"/api/students/{sid}/register-face", headers=h,
                           json={"embeddings": [unit(rng, base, 0.02) for _ in range(n)], "model": model})

    assert reg(alice, alice_base, "Facenet").status_code == 400      # wrong model refused
    assert reg(alice, alice_base, n=2).status_code == 400            # too few
    assert reg(alice, alice_base).status_code == 200
    assert reg(bob, bob_base).status_code == 200
    assert reg(bob, alice_base).status_code == 409                   # same face, other student

    listing = {s["full_name"]: s for s in client.get("/api/students", headers=h).get_json()["students"]}
    assert listing["Alice Ade"]["face_status"] == "registered" and listing["Alice Ade"]["face_registered"]
    assert listing["Old Timer"]["face_status"] == "needs_reregistration" and not listing["Old Timer"]["face_registered"]

    course = db.courses.insert_one({"course_code": "CSC1", "course_title": "X", "is_active": True}).inserted_id
    s = client.post("/api/attendance/start", headers=h, json={"course_id": str(course)})
    assert s.status_code in (200, 201), s.get_json()
    sid = s.get_json()["session"]["id"]

    def rec(emb, model=MODEL):
        Config.RECOGNITION_MIN_INTERVAL_MS = 0
        return client.post("/api/attendance/recognize", headers=h,
                           json={"session_id": sid, "embedding": emb, "model": model}).get_json()

    r = rec(unit(rng, alice_base, 0.02))
    assert r["recognized"] and r["student"]["full_name"] == "Alice Ade" and not r["duplicate"]
    assert rec(unit(rng, alice_base, 0.02))["duplicate"] is True
    assert not rec(unit(rng))["recognized"]                           # stranger
    assert not rec(unit(rng, alice_base, 0.02), "Facenet")["recognized"]   # wrong model, safe message
    assert not rec([0.1] * 5)["recognized"]
    assert db.attendance_records.count_documents({}) == 1

    # legacy student is never matched, and stays in the DB until re-registered
    assert db.face_data.count_documents({"student_id": ObjectId(old), "model": "Facenet"}) == 1
    assert reg(old, unit(rng)).status_code == 200
    assert db.face_data_legacy.count_documents({"student_id": ObjectId(old)}) == 1
    assert db.face_data.count_documents({"student_id": ObjectId(old), "model": "Facenet"}) == 0
    st = client.get(f"/api/students/{old}", headers=h).get_json()["student"]
    assert st["face_status"] == "registered"


def test_import_graph_has_no_heavy_libs():
    import sys
    for heavy in ("tensorflow", "deepface", "keras", "cv2", "retinaface"):
        assert heavy not in sys.modules, heavy
