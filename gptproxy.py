import os
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv


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