from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from uvicorn import run
from router.mirror_router import router as mirror_router
# from yolo.yolo_pose_app import YoloPoseApp
import os

def create_app() -> FastAPI:
    application = FastAPI(
        title="Fast Mirror",
        description="Miró is fast",
    )
    global current_clothe
    global current_pants
    application.include_router(mirror_router)
    
    # CORS 設定
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 或指定你的前端網址，如 ["http://localhost:3000"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return application

app = create_app()
mcp = FastApiMCP(app)
mcp.mount()

if __name__ == "__main__":
    run('main:app', host="0.0.0.0", port=8000, reload=True)
