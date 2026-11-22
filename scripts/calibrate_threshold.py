"""Calibrate FACE_DISTANCE_THRESHOLD from real student photos.

Folder layout (one folder per student, 3+ photos each, ideally 10+):
    photos/
      2021-129965-reg/  a.jpg b.jpg c.jpg ...
      2022-100234-reg/  ...

Run from the backend folder:
    python scripts/calibrate_threshold.py photos
    python scripts/calibrate_threshold.py photos --max-far 0.001

Photos go through the SAME pipeline as live recognition (policy "single"), so
the numbers reflect what the system will really do. Nothing is stored.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("FLASK_DEBUG", "1")  # allow running without production secrets
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from services.face_recognition import FaceError, extract_embedding
from services.face_matching import normalize

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_embeddings(root):
    people = {}
    rejected = 0
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        vectors = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                import cv2

                image = cv2.imread(str(path))
                if image is None:
                    raise FaceError("unreadable")
                vectors.append(normalize(extract_embedding(image, policy="single")))
            except FaceError as exc:
                rejected += 1
                print(f"  skipped {folder.name}/{path.name}: {exc}")
        if len(vectors) >= 2:
            people[folder.name] = np.stack(vectors)
        else:
            print(f"  ignoring {folder.name}: needs at least 2 usable photos")
    return people, rejected


def distance_sets(people):
    genuine = []
    for matrix in people.values():
        d = 1.0 - matrix @ matrix.T
        genuine.extend(d[np.triu_indices(len(matrix), k=1)])

    impostor = []
    for a, b in itertools.combinations(people.values(), 2):
        impostor.extend((1.0 - a @ b.T).ravel())
    return np.array(genuine), np.array(impostor)


def describe(name, values):
    q = np.percentile(values, [0, 5, 50, 95, 100])
    print(f"{name:<9} n={len(values):<7} min={q[0]:.3f} p5={q[1]:.3f} "
          f"median={q[2]:.3f} p95={q[3]:.3f} max={q[4]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--max-far", type=float, default=0.001,
                        help="highest acceptable false-accept rate (default 0.1%%)")
    args = parser.parse_args()

    people, rejected = load_embeddings(args.folder)
    if len(people) < 2:
        sys.exit("Need at least 2 students with 2+ usable photos each.")
    genuine, impostor = distance_sets(people)

    print(f"\n{len(people)} students, {rejected} photos rejected by the quality gate")
    describe("same", genuine)
    describe("different", impostor)

    print("\nthreshold   FAR (wrong person accepted)   FRR (right person rejected)")
    rows = []
    for t in np.round(np.arange(0.20, 0.61, 0.025), 3):
        far = float((impostor < t).mean())
        frr = float((genuine >= t).mean())
        rows.append((t, far, frr))
        print(f"  {t:.3f}      {far * 100:8.3f}%                  {frr * 100:8.3f}%")

    safe = [r for r in rows if r[1] <= args.max_far]
    print()
    if not safe:
        print("No threshold meets the FAR limit with this data. Collect better photos "
              "(frontal, well lit, varied days) or lower the limit knowingly.")
        return
    best = max(safe, key=lambda r: r[0])  # loosest threshold that is still safe
    print(f"Recommended FACE_DISTANCE_THRESHOLD = {best[0]:.3f}  "
          f"(FAR {best[1] * 100:.3f}%, FRR {best[2] * 100:.2f}%)")
    gap = float(np.percentile(impostor, 1)) - best[0]
    print(f"Margin between threshold and the 1st percentile of impostors: {gap:.3f}")
    print("Set FACE_AMBIGUITY_MARGIN to about that gap or 0.05, whichever is smaller.")
    print("Re-run whenever the camera, lighting or pipeline changes.")


if __name__ == "__main__":
    main()
