from fastapi import APIRouter, Request, Body
from tools import parse_starlette_request
from yolo.yolo_pose_app import create_yolo_app
import os

router = APIRouter(
    prefix=f'',
    tags=['Mirror'],
)

@router.post("/yolo/clothes_path")
async def set_clothes_path(request: Request, data: dict = Body(...)):
    """
    設定 YOLO 虛擬試衣的衣服圖路徑
    { "clothes_path": "/home/icam-540/fast_mirror/yolo/tshirt2.png" }
    """
    clothes_path = data.get("clothes_path")
    if not clothes_path or not os.path.exists(clothes_path):
        return {"success": False, "msg": "clothes_path 不存在"}
    app = request.app
    yolo_app = getattr(app.state, "yolo_app", None)
    if yolo_app is None:
        yolo_app = create_yolo_app(clothes_path)
        app.state.yolo_app = yolo_app
        return {"success": True, "msg": f"YOLO App 尚未初始化，已建立並套用衣服圖 {clothes_path}"}
    yolo_app.set_clothes_path(clothes_path)
    return {"success": True, "msg": f"已更換衣服圖為 {clothes_path}"}

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
