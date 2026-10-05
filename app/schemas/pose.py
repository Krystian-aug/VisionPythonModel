from typing import List

from pydantic import BaseModel

class Keypoint(BaseModel):
    """Pojedynczy punkt ciała wykryty przez model."""

    name: str
    x: float
    y: float
    confidence: float

class PoseResult(BaseModel):
    """Wynik analizy pozy dla konkretnego ćwiczenia."""

    exercise: str
    view: str
    keypoints: list[Keypoint]