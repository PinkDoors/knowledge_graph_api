# app/models.py
from pydantic import BaseModel
from typing import List

class DescriptionRequest(BaseModel):
    text: str

class ConceptEdge(BaseModel):
    node_1: str
    node_2: str
    edge: str

# Response is simply a list of ConceptEdge
ConceptGraphResponse = List[ConceptEdge]