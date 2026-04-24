import numpy as np
from fastapi import FastAPI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.helpers.df_helpers import df2Graph, graph2Df, documents2Dataframe
from app.helpers.prompts import quizPrompt, tutorPrompt, tutorPromptWithGraph, tutorPromptV2, tutorPromptWithGraphV2
from app.models.ask import AskResponse, AskRequest, AskWithGraphRequest
from app.models.models import DescriptionRequest, ConceptEdge, ConceptGraphResponse
from app.models.quiz import QuizRequest, QuizResponse, QuizQuestion
from app.validation.quiz_validator import validate_question

app = FastAPI(title="Text to Graph API")

@app.post("/generate_graph", response_model=ConceptGraphResponse)
def generate_graph(request: DescriptionRequest):
    text = request.text

    print(f"Accepted generate request for text: {text}")

    # Wrap API string in a LangChain Document
    documents = [Document(page_content=text, metadata={})]

    # Split into pages/chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    pages = splitter.split_documents(documents)

    # Convert chunks to dataframe
    df = documents2Dataframe(pages)

    # Extract concepts using your helper functions
    concepts_list = df2Graph(df, model="deepseek-v3.1:671b-cloud")
    dfg = graph2Df(concepts_list)

    # Clean dataframe
    dfg.replace("", np.nan, inplace=True)
    dfg.dropna(subset=["node_1", "node_2", "edge"], inplace=True)

    # Convert to JSON-friendly format
    response = [
        ConceptEdge(
            node_1=str(row["node_1"]),
            node_2=str(row["node_2"]),
            edge=str(row["edge"])
        )
        for _, row in dfg.iterrows()
    ]

    return response

@app.post("/generate_quiz_questions", response_model=QuizResponse)
def generate_quiz_questions(request: QuizRequest):

    graph_data = [
        {
            "node_1": e.node_1,
            "node_2": e.node_2,
            "edge": e.edge
        }
        for e in request.graph
    ]

    raw_questions = quizPrompt(graph_data, request.num_questions)

    if not raw_questions:
        return QuizResponse(questions=[])

    validated_questions = []

    for q in raw_questions:
        try:
            validate_question(q)
            validated_questions.append(QuizQuestion(**q))
        except Exception as e:
            print("Skipping invalid question:", q)

    return QuizResponse(questions=validated_questions)

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    print(request.userPrompt)
    print(request.knowledge)
    # Convert map -> list for prompt readability
    knowledge_list = [{"concept": c, "mastery": m} for c, m in request.knowledge.items()]

    # Use tutorPrompt directly
    response_text = tutorPromptV2(request.userPrompt, knowledge_list)

    return AskResponse(response=response_text)

@app.post("/ask-with-graph", response_model=AskResponse)
def ask_with_graph(request: AskWithGraphRequest):
    print(request.userPrompt)
    print(request.knowledge)
    print(request.graphEdges)

    knowledge_list = [
        {"concept": c, "mastery": m}
        for c, m in request.knowledge.items()
    ]

    graph_list = [
        {
            "from": e.node_1,
            "to": e.node_2,
            "relation": e.edge
        }
        for e in request.graphEdges
    ]

    response_text = tutorPromptWithGraphV2(
        request.userPrompt,
        knowledge_list,
        graph_list
    )

    return AskResponse(response=response_text)