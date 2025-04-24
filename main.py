from fastapi import FastAPI
from uvicorn import run
from router.mirror_router import router as mirror_router
from yolo.yolo_pose_app import YoloPoseApp
import os

def creat_app() -> FastAPI:
    application = FastAPI(
        title="Fast Mirror",
        description="Miró is fast",
    )
    
    application.include_router(mirror_router)
    
    return application

app = creat_app()

# 啟動時初始化 YoloPoseApp 並存到 app.state
def on_startup():
    app.state.yolo_app = YoloPoseApp(
        model_path=os.path.expanduser("~/fast_mirror/yolo/yolo11m-pose.pt"),
        clothes_path=os.path.expanduser("~/fast_mirror/yolo/tshirt1.png"),
        camera_index=10,
        stream_size=(1080, 1920),
        stream_quality=75,
        stream_fps=30
    )

app.add_event_handler("startup", on_startup)

if __name__ == "__main__":
    run('main:app', host="0.0.0.0", port=8000, reload=True)
