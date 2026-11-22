# Optimization notes

## 0. Read this first: Python 3.14

Your `__pycache__` files are `cpython-314`. TensorFlow (which DeepFace/Facenet needs) did not publish
Python 3.14 wheels as of my search: `pip install tensorflow` fails there with "No matching distribution".
If face registration works for you today, your venv is probably on a different Python. If not, create a
**Python 3.11, 3.12 or 3.13** venv for this backend. Nothing in this code needs 3.14.

## 1. What was slow (findings)

| # | Problem (verified in your code) | Severity | Fix | Expected gain |
|---|---|---|---|---|
| 1 | `identify_student` read every embedding from MongoDB and looped in Python per frame | High | In-memory NumPy matrix, one matrix-vector product | 50x at 100 samples, ~220x at 10,000 (measured, see section 4), plus no per-frame DB read |
| 2 | Face detected twice (Haar, then DeepFace "opencv") | High | Detect once, crop, send the crop to Facenet with `detector_backend="skip"` | Roughly halves detection cost |
| 3 | Inactive student's embedding could win, then be discarded, hiding the right student | High (correctness) | Only active students are loaded into the index | n/a |
| 4 | `debug=True` reloader imported the app twice (two index passes, possible two model loads); bound to 0.0.0.0 | Medium | Reloader off by default, host 127.0.0.1, both configurable | Half the start-up work, less RAM |
| 5 | Model loaded lazily on the first real frame | Medium | Background warm-up at start-up | First scan no longer hangs for seconds |
| 6 | Haar ran on full-resolution frames | Medium | Detect on a copy with longest side 640 px | Large on 1080p webcams |
| 7 | `/recognize` did 3+ DB reads per frame, no overlap or rate control | Medium | Short-TTL session/access cache, busy flag, min interval, "already marked" memory | Zero DB reads on repeat sightings |
| 8 | `course_report` loaded all records into Python; export built the CSV in memory | Medium | `$group` aggregation; streamed CSV | Memory stays flat |
| 9 | Indexes created from five route modules; missing indexes for report queries | Low/Med | One `ensure_indexes()`; new indexes justified per query | Faster reports |
| 10 | Race: two "start session" requests could both pass the check | Low/Med | Partial unique index + `DuplicateKeyError` handling | n/a |
| 11 | `register_face` deleted old faces before inserting new ones | Medium | Insert new, then delete old | No window with no face |
| 12 | `cv2.imencode` result unused; password hash fetched on every request | Low | Removed; projection excludes `password_hash` | Small |

## 2. File-by-file changes

New: `utils.py`, `database/indexes.py`, `services/face_matching.py` (pure NumPy), `services/face_cache.py`,
`services/session_state.py`, `serve.py`, `scripts/*`, `tests/*`, `.env.example`.
Rewritten: `config.py`, `database/db.py`, `app.py`, `services/face_recognition.py`, `services/attendance_service.py`,
`services/auth_service.py`, and all of `routes/*`.
`database/__init__.py`, `routes/__init__.py`, `services/__init__.py` stay empty as before.

### Behaviour you should know about (API compatibility)

All endpoints, methods and existing response fields are unchanged. Additions only:

* `/api/attendance/recognize` may now return `{"recognized": false, "busy": true, ...}` (HTTP 200) when a previous
  frame is still processing or frames arrive faster than `RECOGNITION_MIN_INTERVAL_MS`. Existing frontends already treat
  `recognized: false` as "keep scanning".
* Frames with two similar-sized faces, an unclear match (two students too close) or poor quality return
  `recognized: false` with a message.
* `distance` is still returned. Set `RETURN_DISTANCE=0` once your React code does not read it.
* List endpoints (`/students`, `/attendance`, `/reports/attendance`) now also return `page`, `limit`, `total`, `pages`.
  `/students` defaults to 200 per page (same as before); use `?page=2` for more.
* Error bodies are always JSON `{"message": ...}`. Database outages return 503 instead of a crash.
* `/api/health` now also reports `model` state and face-cache counts (no names, no embeddings).

## 3. Migration notes (do these once)

1. **Use Python 3.11-3.13** (section 0) and `pip install -r requirements.txt` plus `requirements-additions.txt` (adds `waitress`).
2. **Set real secrets in `.env`**: `SECRET_KEY` and `JWT_SECRET`. With `FLASK_DEBUG=0` (the default now) the app refuses
   to start on the `dev-only-change-me` defaults. For local development set `FLASK_DEBUG=1`.
3. **Existing faces: re-register them.** The old pipeline sent the whole frame to DeepFace, which detected and aligned
   its own way. The new one crops and aligns itself. Old embeddings still load and match (they are normalised on load),
   but crops differ slightly, so accuracy is best when everyone is re-registered. `/api/health` shows
   `face_cache.legacy_samples` = how many old samples remain.
4. **Indexes** are created automatically at start-up. If two sessions are already `active` for one course, the new
   "one active session per course" index is skipped with a warning in the log. End the extra session and restart.
5. A small `meta` collection (document `face_data_version`) is created automatically.
6. Run the backend as **one process**: `python serve.py` (waitress, 4 threads) or `python app.py` for development.
   Multiple worker processes would each load their own copy of the model (hundreds of MB) and keep their own caches
   and busy/rate-limit state. Threads share all of it, and a lock serialises TensorFlow inference.

## 4. Benchmarks

Matching one frame against N stored samples (median ms, measured here on a cloud CPU, synthetic 128-d vectors):

| samples | old loop | new matrix | speed-up |
|---|---|---|---|
| 100 | 0.63 | 0.012 | 51x |
| 1,000 | 7.1 | 0.039 | 180x |
| 10,000 | 69.9 | 0.32 | 219x |

The old figures exclude MongoDB: the old code also pulled every document over the network per frame, so real-world
gains are larger. **Not measured here:** embedding time and `/recognize` end to end, because TensorFlow/DeepFace cannot be
installed in this environment and I had no real face photos. Run on your laptop:

```
python scripts/benchmark.py
python scripts/benchmark.py --image my_face.jpg
python scripts/benchmark.py --url http://127.0.0.1:5000 --token <JWT> --session <SESSION_ID> --image my_face.jpg
```

## 5. Calibrating the threshold with real student photos

The threshold stays at **0.40** (`FACE_DISTANCE_THRESHOLD`); I did not loosen it. To check it for your cameras:

1. Collect 10+ photos per student for at least 20-30 students, on the real attendance webcam, in the real room lighting,
   on different days. Put them in `photos/<matric>/*.jpg` (use `2021-129965-reg` style folder names).
2. `python scripts/calibrate_threshold.py photos --max-far 0.001`
3. It prints the distance spread for the same person vs different people and the false-accept / false-reject rate at each
   threshold, then recommends the loosest threshold that keeps false accepts under your limit.
4. Put the result in `.env`. Prefer a low false-accept rate over a low false-reject rate: a student who is rejected just
   scans again; a wrongly accepted face records someone else's attendance.
5. Re-run after changing camera, lighting or any `FACE_*` setting.

## 6. Test checklist

Automated (`python -m pytest`, 28 tests): matching, unknown face, threshold edges, ambiguity, many samples per student,
inactive students excluded, registration conflict detection, same-person consistency, legacy/NaN/odd-size records,
overlap blocking, rate limiting, session cache expiry, oversized/garbage images.

Manual, with MongoDB and a camera:

* Register a student (3-8 photos): succeeds; photos of two different people in one request are rejected; a second
  student using the same face gets 409.
* Start a session; scan the registered student: `recognized: true`. Scan again: `duplicate: true`, and no new DB record.
* Unknown face: `recognized: false`. Two people side by side: rejected as more than one face. Cover the lens or turn
  off the light: rejected as poor lighting / too blurry.
* Two similar-looking students: the match is rejected as unclear rather than guessing.
* Deactivate a student mid-session: their next scan is "Unknown face" within about 2 seconds.
* End the session, then scan: 409 "not active". Start two sessions for one course at once: second gets 409.
* Stop MongoDB: API returns 503 JSON; the app recovers when it comes back.

## 7. Frontend note (React, no API change needed)

* Send one frame every 1.0-1.5 s, and **do not send the next one until the previous response arrives** (use a loop with
  `await`, not `setInterval`).
* Draw the video frame to a canvas no larger than about 640 px on the longest side and send
  `canvas.toDataURL("image/jpeg", 0.7)`. Faces still need to be at least 80 px wide in that image.
* If the response has `busy: true`, simply wait and send the next frame later.
* For registration, ask for varied but frontal poses, in good light.

## 8. Deliberately not changed

* **Haar detector kept** (no new dependency, CPU-cheap). It is less accurate at angles than modern detectors; the quality
  gate and the consistency check on registration compensate. If you see many misses, a lightweight detector such as
  YuNet (shipped inside OpenCV) is the next step, but it changes registration crops, so re-register.
* **Facenet kept** (as you asked); `PIPELINE_VERSION` is stored with each embedding so a later model change can be detected.
* **No Redis, vector DB, Celery, WebSockets or GPU.** A brute-force NumPy search over thousands of students is well under a
  millisecond; a vector database would only add complexity.
* **JWT is still checked and the user still read from MongoDB on every request**, so a deactivated account is locked out
  immediately. Only session and course-assignment lookups are cached (30 s), so changing a lecturer's course assignment
  can take up to 30 s to affect a scan already in progress (assignment changes clear the cache immediately in the same process).
* **No rate limiting on `/login`** and no token revocation list (logout is client side). Worth adding before public deployment.
* **Not validated in this environment:** the live Haar + Facenet path and Mongo-backed routes (no TensorFlow, no MongoDB,
  no real face photos here). The pure matching/cache/session logic, image decoding, app start-up and error handling were
  run and tested. Please do the manual checklist above on your machine before relying on it.
