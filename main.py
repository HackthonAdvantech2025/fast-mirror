from fastapi import FastAPI
from uvicorn import run
from router.mirror_router import router as mirror_router

def creat_app() -> FastAPI:
    application = FastAPI(
        title="Fast Mirror",
        description="Miró is fast",
    )
    
    application.include_router(mirror_router)
    
    return application

app = creat_app()

if __name__ == "__main__":
    run('main:app', host="0.0.0.0", port=8000, reload=True)
