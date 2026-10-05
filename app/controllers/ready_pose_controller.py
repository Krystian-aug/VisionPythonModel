from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.pose import PoseResult
from app.services.ready_pose_service import ReadyPoseService


router = APIRouter(
    prefix="/api/ready-pose",
    tags=["Ready Pose"]
)

service = ReadyPoseService()


@router.get("/health")
def health():
    """Sprawdza, czy gotowy model pose estimation jest dostępny."""

    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Ready pose model is not ready"
        )

    return {
        "status": "UP",
        "model": "yolo11n-pose",
        "message": "Ready pose model is ready"
    }


@router.post("", response_model=PoseResult)
async def predict_pose(
    image: UploadFile = File(...),
    exercise: str = "squat"
):
    """Przyjmuje zdjęcie i nazwę ćwiczenia, a następnie uruchamia analizę pozy."""

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported"
        )

    try:
        image_bytes = await image.read()

        return service.predict(
            image_bytes=image_bytes,
            exercise=exercise
        )

    except ValueError as exception:
        raise HTTPException(
            status_code=400,
            detail=str(exception)
        )