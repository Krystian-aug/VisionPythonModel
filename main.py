from fastapi import FastAPI

from app.controllers.ready_pose_controller import router as ready_pose_router


app = FastAPI(
    title="VisionPythonModel",
    description="API for human pose estimation",
    version="1.0.0"
)

app.include_router(ready_pose_router)


@app.get("/")
def root():
    return {
        "message": "VisionPythonModel API is running"
    }