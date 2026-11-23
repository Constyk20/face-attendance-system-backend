"""Match one face embedding (computed in the browser) to a registered student.

No image processing and no neural network here: validation, then one
matrix-vector product against the in-memory index of enrolled embeddings.
"""

from dataclasses import dataclass

from services.face_cache import face_index
from services.face_recognition import validate_embedding


@dataclass
class Outcome:
    status: str  # "match" | "unknown" | "ambiguous"
    student: dict = None  # {"id": ObjectId, "full_name": str, "matric_number": str}
    distance: float = None
    message: str = ""


def identify_embedding(embedding_value, model):
    """Validate and match. Raises FaceError for malformed data or a model mismatch."""
    embedding = validate_embedding(embedding_value, model)

    snapshot, result = face_index.match(embedding)
    if result.status == "match":
        return Outcome(
            "match",
            student=snapshot.labels[result.index],
            distance=round(result.distance, 4),
        )
    if result.status == "ambiguous":
        return Outcome(
            "ambiguous",
            message="Face match is not clear. Look straight at the camera and try again",
        )
    return Outcome("unknown", message="Unknown face")
