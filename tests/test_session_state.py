from services.session_state import SessionState


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(**kw):
    clock = Clock()
    state = SessionState(
        ttl=kw.get("ttl", 30),
        min_interval_ms=kw.get("interval", 700),
        max_pending=kw.get("pending", 4),
        clock=clock,
    )
    return state, clock


def test_overlap_is_rejected_then_allowed_after_finish():
    s, clock = make()
    assert s.begin("x") == "ok"
    assert s.begin("x") == "busy"
    s.finish("x")
    clock.now += 1
    assert s.begin("x") == "ok"


def test_minimum_interval_throttles():
    s, clock = make(interval=700)
    assert s.begin("x") == "ok"
    s.finish("x")
    clock.now += 0.2
    assert s.begin("x") == "throttled"
    clock.now += 0.6
    assert s.begin("x") == "ok"


def test_global_pending_limit():
    s, _ = make(pending=2)
    assert s.begin("a") == "ok"
    assert s.begin("b") == "ok"
    assert s.begin("c") == "busy"
    s.finish("a")
    assert s.begin("c") == "ok"


def test_marked_students_skip_repeat_work():
    s, _ = make()
    assert s.marked("x", "stu") is None
    s.mark("x", "stu", {"id": "rec"})
    assert s.marked("x", "stu") == {"id": "rec"}
    s.drop_session("x")
    assert s.marked("x", "stu") is None


def test_session_cache_expires_and_invalidates():
    s, clock = make(ttl=30)
    s.store_session("x", {"course_id": "c1", "status": "active"})
    assert s.cached_session("x")["status"] == "active"
    clock.now += 31
    assert s.cached_session("x") is None
    s.store_session("x", {"course_id": "c1", "status": "active"})
    s.drop_session("x")
    assert s.cached_session("x") is None


def test_drop_course_clears_sessions_and_access():
    s, _ = make()
    s.store_session("x", {"course_id": "c1"})
    s.store_access("u1", "c1", True)
    s.drop_course("c1")
    assert s.cached_session("x") is None
    assert s.access("u1", "c1") is None


def test_dropping_busy_session_releases_global_slot():
    s, _ = make(pending=1)
    assert s.begin("a") == "ok"
    s.drop_session("a")
    assert s.begin("b") == "ok"
