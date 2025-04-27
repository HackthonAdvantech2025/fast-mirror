import os
# fastapi
from fastapi import APIRouter, Request, Body, HTTPException
from typing import List

from gptproxy import chat_handler
# repo
from repository.mirror_repo import get_product_info
from repository.bedrock_repo import chat_with_nova
# model
from model.mirror_model import MessageContent

from tools import parse_starlette_request
import httpx
import random

router = APIRouter(
    prefix=f'',
    tags=['Mirror'],
)

@router.post('/chat')
async def chat_handler(
        request: Request,
        message_log: List[MessageContent] = Body(...),
        operation_id='chat_handler',
    ):
    response: MessageContent = await chat_with_nova(message_log)

    return response

# 判斷是否有衣服路徑
#   有: set_clothes
#   無: find_clothes_path
#       取得推薦衣服列表: clothes_recommend_list
#       根據當前衣服id找在list中的哪個衣服，再根據是prev還是next取得衣服路徑

# @router.get("/find_clothes_path")
# async def find_clothes_index(request: Request, operation_id='find_clothes_path'):
#     """
#     取得衣服路徑
#     """
#     clothes_list = await get_product_info()
    
#     current_clothes_rul = current_clothes.get('s3url')

#     for index, item in enumerate(clothes_list_cache):
#         item_url = item.get('s3url')
#         if item_url and item_url == current_clothes_rul:
#             return index
    

@router.get("/get_recommend_list")
async def get_recommend_list(request: Request, operation_id='get_recommend_list'):
    """
    取得推薦衣服列表
    """
    response = await get_product_info()
    return response

# 隨機決定推薦衣服
@router.get("/recommend_clothes")
async def recommend_clothes(request: Request, operation_id='recommend_clothes'):
    """
    隨機決定推薦衣服
    """
    response = await get_product_info()
    random_clothes = random.choice(response)
    current_clothes = random_clothes
    return current_clothes


@router.post("/get_prev_clothes_info")
async def get_prev_clothes_info(request: Request, clothe_info: dict = Body(...), operation_id='get_prev_clothes_path'):
    """
    取得上一件衣服資訊
    """
    index = await find_clothes_index(request)
    clothes_list = await get_product_info()
    
    if index > 0:
        current_clothes = clothes_list[index - 1]
    else:
        current_clothes = clothes_list[len(clothes_list) - 1]
        
    return current_clothes

@router.post("/get_next_clothes_info")
async def get_next_clothes_info(request: Request, clothe_info: dict = Body(...), operation_id='get_next_clothes_path'):
    """
    取得下一件衣服資訊
    """
    
    index = await find_clothes_index(request)
    clothes_list = await get_product_info()
    
    if index < len(clothes_list) - 1:
        current_clothes = clothes_list[index + 1]
    else:
        current_clothes = clothes_list[0]
    
    return current_clothes

@router.post("/yolo/clothes")
async def call_yolo_set_clothes(request: Request, clothes_info: dict = Body(...), operation_id='set_clothes'):
    """
    接收衣服資訊，並將其轉發給 YOLO 服務設定衣服路徑。
    """
    target_url = "http://192.168.0.119:8080/set_clothes_path"
    payload = {
        'clothes_path': clothes_info.s3Url,
        'part': clothes_info.category
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(target_url, json=payload)
            response.raise_for_status()
            return {"status": "success", "message": "Request sent to YOLO service.", "yolo_response": response.json()}
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error requesting YOLO service: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"YOLO service returned an error: {exc.response.text}")