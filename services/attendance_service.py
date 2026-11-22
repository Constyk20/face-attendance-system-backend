"""Match one webcam frame to a registered student (no database work per frame)."""

from dataclasses import dataclass

from services.face_cache import face_index
from services.face_recognition import decode_image, extract_embedding


@dataclass
class Outcome:
    status: str  # "match" | "unknown" | "ambiguous"
    student: dict = None  # {"id": ObjectId, "full_name": str, "matric_number": str}
    distance: float = None
    message: str = ""


def identify_frame(image_value):
    """Decode, embed and match. Raises FaceError for unusable images."""
    image = decode_image(image_value)
    # "dominant": use the largest face, reject frames with two similar-sized faces.
    embedding = extract_embedding(image, policy="dominant")

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
