from app.schemas.pose import Keypoint


class PoseMapper:
    """Mapuje keypointy YOLO na punkty anatomiczne używane przez aplikację."""

    YOLO_KEYPOINTS = {
        "shoulder": {
            "left": 5,
            "right": 6
        },
        "wrist": {
            "left": 9,
            "right": 10
        },
        "hip": {
            "left": 11,
            "right": 12
        },
        "knee": {
            "left": 13,
            "right": 14
        },
        "ankle": {
            "left": 15,
            "right": 16
        }
    }

    def map_keypoints(self, result, required_keypoints: list[str]) -> list[Keypoint]:
        """Wybiera widoczną stronę ciała i zwraca wymagane punkty."""

        side = self._select_side(
            result,
            required_keypoints
        )

        return self._extract_keypoints(
            result,
            required_keypoints,
            side
        )

    def _select_side(
        self,
        result,
        required_keypoints: list[str]
    ) -> str:
        """Wybiera stronę ciała na podstawie średniej wartości confidence."""

        left_confidence = 0.0
        right_confidence = 0.0

        for keypoint_name in required_keypoints:
            indices = self.YOLO_KEYPOINTS[keypoint_name]

            left_index = indices["left"]
            right_index = indices["right"]

            left_confidence += result.keypoints.conf[0][left_index].item()
            right_confidence += result.keypoints.conf[0][right_index].item()

        if left_confidence >= right_confidence:
            return "left"

        return "right"

    def _extract_keypoints(
        self,
        result,
        required_keypoints: list[str],
        side: str
    ) -> list[Keypoint]:
        """Pobiera współrzędne wymaganych punktów dla wybranej strony ciała."""

        keypoints = []

        for name in required_keypoints:
            index = self.YOLO_KEYPOINTS[name][side]

            x, y = result.keypoints.xy[0][index].tolist()
            confidence = result.keypoints.conf[0][index].item()

            keypoints.append(
                Keypoint(
                    name=name,
                    x=x,
                    y=y,
                    confidence=confidence
                )
            )

        return keypoints