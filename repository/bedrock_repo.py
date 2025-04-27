import boto3
import json
from botocore.config import Config
from model.mirror_model import MessageContent
from typing import List

async def chat_with_nova(message_log: List[MessageContent],clothe_info):
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
        return {"error": str(e)}
    

