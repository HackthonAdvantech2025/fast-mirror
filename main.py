from fastapi import FastAPI
from uvicorn import run
from router.test_router import router as test_router

def create_app() -> FastAPI:
    application = FastAPI(
        title="Fast Mirror",
        description="Miró is fast",
    )
    
    application.include_router(test_router)
    
    return application

app = create_app()

if __name__ == "__main__":
    run('main:app', host="0.0.0.0", port=8000, reload=True)
