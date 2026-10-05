from io import BytesIO

from PIL import Image
from ultralytics import YOLO

from app.config.exercise_config import EXERCISE_CONFIG
from app.schemas.pose import PoseResult
from app.services.pose_mapper import PoseMapper


class ReadyPoseService:
    """Serwis odpowiedzialny za analizę zdjęć przy użyciu gotowego modelu YOLO Pose."""

    def __init__(self):
        """Ładuje model YOLO Pose oraz przygotowuje mapper keypointów."""

        self.model = YOLO("yolo11n-pose.pt")
        self.mapper = PoseMapper()

    def is_ready(self) -> bool:
        """Sprawdza, czy model został poprawnie załadowany."""

        return self.model is not None

    def predict(
        self,
        image_bytes: bytes,
        exercise: str
    ) -> PoseResult:
        """Analizuje zdjęcie i zwraca punkty wymagane dla danego ćwiczenia."""

        if exercise not in EXERCISE_CONFIG:
            raise ValueError(
                f"Unsupported exercise: {exercise}"
            )

        exercise_config = EXERCISE_CONFIG[exercise]

        image = Image.open(
            BytesIO(image_bytes)
        )

        results = self.model(image)

        keypoints = self.mapper.map_keypoints(
            results[0],
            exercise_config["keypoints"]
        )

        return PoseResult(
            exercise=exercise,
            view=exercise_config["view"],
            keypoints=keypoints
        )