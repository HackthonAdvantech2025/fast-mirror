import os
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv

import boto3
import json
from botocore.config import Config
from model.mirror_model import MessageContent
from typing import List

async def chat_handler(message: str):
    load_dotenv()

    ENDPOINT = 'https://models.github.ai/inference'
    MODEL = 'openai/gpt-4.1'
    TOKEN = os.getenv('GITHUB_TOKEN')

    if not TOKEN:
        print("錯誤：請在 .env 檔案中設定 GITHUB_TOKEN 環境變數。")
        exit(1)
    
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "model": MODEL,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ENDPOINT}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
    if response.status_code != 200:
        detail = response.json().get("error", {}).get("message", "Unknown error")
        raise HTTPException(status_code=500, detail=f"GPT Error: {detail}")
    data = response.json()
    print('data: ', data)
    try:
        result = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="GPT response format error")
    return result

async def chat_with_nova(message_log: List[MessageContent]):
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2",
            config=Config(connect_timeout=5, read_timeout=60)
        )
        formatted_messages = []
        for msg in message_log:
            formatted_messages.append({
                "role": msg.role,
                "content": [{"text": item.text} for item in msg.content] # 假設 MessageContent 的 content 是一個包含 text 屬性的物件列表
            })

        response = bedrock_runtime.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=formatted_messages,
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.5,
                "topP": 0.9,
                "stopSequences": []
            }
        )
        print(f"response: {json.dumps(response, indent=2)}")
        return {"response": response['output']['message']['content']}
    except Exception as e:
        print(f"Error in chat_with_nova: {e}, lineNo. {e.__traceback__.tb_lineno}")
        return {"error": str(e)}

async def main():
    # 測試 chat_with_nova 函數
    message_log = [
        MessageContent(role="user", content=[{"text": "Hello, how are you?"}]),
        MessageContent(role="assistant", content=[{"text": "I'm fine, thank you!"}]),
        MessageContent(role="user", content=[{"text": "What can you do?"}]),
    ]
    response = await chat_with_nova(message_log)
    print(response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())