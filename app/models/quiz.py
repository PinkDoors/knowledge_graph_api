from typing import List, Literal, Union
from pydantic import BaseModel, Field

from app.models.models import ConceptEdge


class QuizQuestion(BaseModel):
    concept: str  # <-- added field

    question: str
    type: Literal[
        "multiple_choice_question",
        "true_false_question",
        "multiple_answers_question"
    ]
    options: List[str]

    # int for single-answer, List[int] for multiple_answers_question
    correct_index: List[int]


class QuizRequest(BaseModel):
    graph: List[ConceptEdge]
    num_questions: int = Field(default=5, ge=1, le=50)


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]