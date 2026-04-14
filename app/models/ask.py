from pydantic import BaseModel
from typing import Dict, List

from app.models.models import ConceptEdge


class AskWithGraphRequest(BaseModel):
    userPrompt: str
    knowledge: Dict[str, int]          # 0–100 mastery
    graphEdges: List[ConceptEdge]

class AskRequest(BaseModel):
    userPrompt: str
    knowledge: Dict[str, int]

class AskResponse(BaseModel):
    response: str