from pydantic import BaseModel
from typing import Dict

class AskRequest(BaseModel):
    userPrompt: str
    knowledge: Dict[str, int]

class AskResponse(BaseModel):
    response: str