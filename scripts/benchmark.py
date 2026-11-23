"""Benchmark the recognition hot path.

Run from the backend folder:
    python scripts/benchmark.py                 # matching (no model: embeddings come from the browser)
    python scripts/benchmark.py --url http://127.0.0.1:5000 --token JWT \
        --session SESSION_ID                    # also times /recognize end to end

"Old" below is the original algorithm (Python loop of per-record cosine distance,
one database read per record). "New" is the NumPy matrix. Database time is NOT
included for the old method, so the real-world gap is larger than shown.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from services.face_matching import build_snapshot, match

DIM = 128  # SFace


def timeit(fn, runs):
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), min(samples)


def old_cosine(left, right):
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denom == 0 else 1.0 - float(np.dot(a, b) / denom)


def old_match(records, query):
    best = 0.40
    best_id = None
    for student_id, stored in records:
        distance = old_cosine(query, stored)
        if distance < best:
            best, best_id = distance, student_id
    return best_id


def bench_matching(samples_per_student=5):
    print("\n== Matching one frame against N enrolled samples (median ms) ==")
    print(f"{'samples':>9} {'old loop':>12} {'new matrix':>12} {'speed-up':>10}")
    rng = np.random.default_rng(0)
    for n in (100, 1_000, 10_000):
        vectors = rng.normal(size=(n, DIM)).astype(np.float32)
        records = [(i // samples_per_student, vectors[i].tolist()) for i in range(n)]
        snapshot = build_snapshot((sid, None, vec) for sid, vec in records)
        query = rng.normal(size=DIM).astype(np.float32)

        runs = 5 if n >= 10_000 else 15
        old_ms, _ = timeit(lambda: old_match(records, query), runs)
        new_ms, _ = timeit(lambda: match(snapshot, query, 0.50, 0.05), 50)
        print(f"{n:>9} {old_ms:>12.3f} {new_ms:>12.4f} {old_ms / new_ms:>9.0f}x")


def bench_endpoint(url, token, session, runs=10):
    rng = np.random.default_rng(1)
    vector = rng.normal(size=DIM)
    vector = (vector / np.linalg.norm(vector)).round(6).tolist()
    body = json.dumps(
        {"session_id": session, "embedding": vector, "model": "sface-yunet-v1"}
    ).encode()

    def call():
        req = urllib.request.Request(
            url.rstrip("/") + "/api/attendance/recognize",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            response.read()

    print("\n== POST /api/attendance/recognize end to end (ms) ==")
    call()  # warm-up
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
        time.sleep(0.8)  # respect RECOGNITION_MIN_INTERVAL_MS
    print(f"median {statistics.median(samples):.0f}  min {min(samples):.0f}  max {max(samples):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--token")
    parser.add_argument("--session")
    args = parser.parse_args()

    bench_matching()
    if args.url and args.token and args.session:
        bench_endpoint(args.url, args.token, args.session)
