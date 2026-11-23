# Browser face recognition: what changed, threshold, re-registration, deployment

## 1. What changed (minimum changes)

| Where | Change |
|---|---|
| Browser (new) | `src/services/faceEngine.js`: YuNet detects the face + 5 landmarks, the face is aligned to 112x112, SFace (int8) makes a 128-number embedding. Runs with ONNX Runtime Web (WASM, 1 thread). Models load once and are cached by the browser. |
| Browser (edited) | `FaceRegister.jsx`, `Attendance.jsx`, `WebcamCapture.jsx`, `api.js`: use embeddings instead of JPEGs. Students/Faces badges and the dashboard status line show the new states. UI design unchanged. |
| Flask | `services/face_recognition.py` now only **validates** embeddings. `attendance_service`, `face_cache`, `routes/students.py`, `routes/attendance.py`, `config.py`, `app.py` adapted. Matching code (`face_matching.py`), auth, roles, sessions, reports, dashboards, health endpoints, routes and the database layout are untouched (additive fields only). |
| Removed | DeepFace, TensorFlow, Keras, retina-face, OpenCV, gdown, model warm-up thread, image decoding. |

API contract (same URLs, same response shapes):

* `POST /api/students/<id>/register-face`  body `{ "embeddings": [[128 numbers] x 3..8], "model": "sface-yunet-v1" }`
* `POST /api/attendance/recognize`  body `{ "session_id", "embedding": [128 numbers], "model": "sface-yunet-v1" }`
* Responses (`recognized`, `student`, `record`, `duplicate`, `busy`, `message`, `distance`) are unchanged.
* `/api/health` still returns `model`; it is now `"browser"`, plus `embedding_model`.

## 2. Threshold (`FACE_DISTANCE_THRESHOLD`, default 0.50)

Distance = 1 - cosine similarity (lower = more similar). The old Facenet value 0.40 is **not** reused.

Measured with the real browser pipeline (headless Chromium, ONNX Runtime Web, int8 SFace) on the 61 test photos / 300 labelled pairs from the DeepFace test set:

| | n | min | mean | max |
|---|---|---|---|---|
| Same person | 38 | 0.108 | 0.238 | 0.397 |
| Different people | 262 | 0.664 | 0.855 | - |

0.50 sits in the gap: 0 false accepts and 0 false rejects on that set. OpenCV's published SFace limit is 0.637, but it leaves little room before false accepts, so a stricter value was chosen. `FACE_AMBIGUITY_MARGIN=0.05` additionally rejects a frame when two different students are almost equally close.

Limits of this check: the set is small (about 40 people, mostly celebrity photos), not your students and your classroom camera. **Re-check on real students**, then adjust only through the env var:
* a student is often "Unknown face" -> raise to 0.55;
* a wrong student is ever marked -> lower to 0.45 and re-register that student in better light.

## 3. Re-registering existing students (DeepFace/Facenet -> SFace)

Facenet and SFace embeddings are both 128 numbers long, so they cannot be told apart by size. Every stored embedding carries `model`; matching loads **only** embeddings with the current model. They are never mixed.

1. Deploy backend + frontend. Existing students are not deleted or changed. Their old samples stay in `face_data` but are ignored, so they will not be recognised until re-registered.
2. Students list and "Register Faces" show an amber **Re-register needed** badge for them (`face_status: "needs_reregistration"`).
3. Run `python scripts/legacy_faces_report.py` (read only) for the list.
4. Admin opens Register Faces -> Re-register, captures 3-5 photos, Save. The new samples are inserted first; the old ones are then **copied to `face_data_legacy`** and removed from `face_data`. Nothing is overwritten silently, and a failure part-way keeps the old data.
5. To roll back: put the old backend/frontend back; the archived documents in `face_data_legacy` can be copied back into `face_data`.

Old photos were never stored, so embeddings cannot be converted: re-registration is required.

## 4. Deploy

**Render Free (backend)** - `render.yaml` is included.
* Start: `gunicorn app:app --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:$PORT`; build: `pip install -r requirements.txt`; health check `/api/health`.
* Env: `SECRET_KEY`, `JWT_SECRET`, `MONGODB_URI` (Atlas; allow Render's outbound IPs or 0.0.0.0/0 in Atlas Network Access), `FRONTEND_ORIGIN` (your Vercel URL, no trailing slash), `FLASK_DEBUG=0`. Everything else has a default (`.env.example`).
* Measured here: fresh venv install, gunicorn worker about 70 MB RSS (+ about 23 MB master) with `/api/health` answering, versus several hundred MB with TensorFlow. No warm-up work at start.
* Render Free sleeps after ~15 min idle; the first request then takes tens of seconds. Open the site a minute before class, or use an external uptime ping.

**Vercel (frontend)**
* `npm install onnxruntime-web@1.31.1` (exact version: the `.wasm` in `public/ort/` must match it).
* Copy `src/`, `public/models`, `public/ort`, `vercel.json` into the project. `VITE_API_URL=https://<your-render-app>.onrender.com/api`.
* `public/` holds ~25 MB (models 10 MB, WASM 14.8 MB, cached `immutable`). Students download the ~10 MB model once per browser; the WASM downloads (about 3.8 MB compressed) when the face pages first open.
* Camera needs HTTPS (Vercel provides it) or localhost.
* Vite also copies the 14.8 MB `.wasm` into `dist/assets/`; it is unused and can be deleted from `dist` before deploying if you want a smaller upload.

## 5. Model sources and integrity

| File | Source | Licence | SHA-256 |
|---|---|---|---|
| `face_detection_yunet_2026may.onnx` | OpenCV Zoo `models/face_detection_yunet` | MIT | `ebafce4e...c0f0` |
| `face_recognition_sface_2021dec_int8.onnx` | OpenCV Zoo `models/face_recognition_sface` | Apache-2.0 | `2b0e941e...8d78a` |
| `ort-wasm-simd-threaded.{wasm,mjs}` | onnxruntime-web 1.31.1 | MIT | `4b884c20...001d` (wasm) |

SFace (published LFW accuracy 99.40% fp32, 99.32% int8). Swapping to the fp32 file (39 MB) needs a new `EMBEDDING_MODEL` tag and re-registration.

## 6. Honest limits

* **Client-side embeddings are trusted by the server.** Anyone who can sign in can post a crafted embedding for a session they are allowed to use. The server checks format, model tag, norm, session/role permissions, one-frame-at-a-time and the matching threshold, but cannot prove a number came from a camera. This is also not an anti-spoofing check (a photo of a face still works). Fine for a classroom tool behind lecturer logins; not for high-stakes access control.
* A very old phone may take about a second per frame; the loop waits for each frame.
* int8 SFace embeddings differ slightly from OpenCV's (mean cosine 0.95 between my browser pipeline and OpenCV on the same photos); the thresholds above were measured on the browser pipeline itself, which is what runs in production.
* Not tested: a real Render deployment, a real Vercel deployment, MongoDB Atlas, Safari/Firefox, real classroom webcams.

## 7. How this was tested

* `pytest`: 42 tests pass (matching, session state, embedding validation, API flow: register, wrong model, too few captures, duplicate face, recognise, duplicate scan, stranger, legacy student never matched, re-registration archives old data, `import app` loads no tensorflow/deepface/cv2).
* Browser vs OpenCV on 61 photos (YuNet + SFace): same detections; embeddings agree at mean cosine 0.95.
* Real end-to-end: built frontend in Chromium with a fake camera feeding a face video + the real Flask app (in-memory MongoDB): registration UI sent 3 embeddings (3.6 KB, no image), another photo of the same person was recognised and marked once (later scans "already marked"), a different person got "Unknown face".
* Fresh virtualenv from `requirements.txt` + gunicorn start (see Render above).
