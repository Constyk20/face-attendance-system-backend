"""List students whose stored faces were made by the OLD model and must be re-registered.

    python scripts/legacy_faces_report.py            # read-only report (default)

Nothing is changed or deleted. Old (Facenet) embeddings keep sitting in
`face_data`, are ignored by matching, and are moved to `face_data_legacy` only
when an admin re-registers that student (see FACE_MIGRATION.md).
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from database.db import get_db


def main():
    db = get_db()
    current = Config.EMBEDDING_MODEL
    by_student = {}
    for doc in db.face_data.find({}, {"student_id": 1, "model": 1}):
        by_student.setdefault(doc["student_id"], Counter())[doc.get("model") or "unknown(legacy)"] += 1

    needs, ok = [], 0
    for student in db.students.find({"is_active": True}, {"full_name": 1, "matric_number": 1, "face_registered": 1}):
        models = by_student.get(student["_id"], Counter())
        if models.get(current):
            ok += 1
        elif student.get("face_registered") or models:
            needs.append((student.get("matric_number", ""), student.get("full_name", ""), dict(models)))

    print(f"Current model: {current}")
    print(f"Active students already on the current model: {ok}")
    print(f"Active students that must re-register: {len(needs)}")
    for matric, name, models in sorted(needs):
        print(f"  {matric:<18} {name:<32} {models}")
    print(f"Archived legacy samples: {db.face_data_legacy.estimated_document_count()}")


if __name__ == "__main__":
    main()
