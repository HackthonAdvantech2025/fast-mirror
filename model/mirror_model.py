from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

class ContentItem(BaseModel):
    text: str

class MessageContent(BaseModel):
    role: str
    content: List[ContentItem]