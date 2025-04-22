from fastapi import APIRouter

router = APIRouter(
    prefix=f'/miro',
    tags=['test'],
)

@router.get("/")
async def root():
    return {"message": "Hello i'm miro."}
