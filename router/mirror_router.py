from fastapi import APIRouter, Request

from tools import parse_starlette_request

router = APIRouter(
    prefix=f'',
    tags=['Mirror'],
)

@router.get("/change")
async def action(request: Request,):
    print(await parse_starlette_request(request))

    return 
    # pass

@router.get("/dress")
async def dress(request: Request, product_id: str):
    pass
@router.post("/dress")
async def dress(request: Request, product_id: str):
    pass

@router.get("/chat")
async def chat(messages: str):
    pass
