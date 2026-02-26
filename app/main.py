import numpy as np
from fastapi import FastAPI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.helpers.df_helpers import df2Graph, graph2Df, documents2Dataframe
from app.models.models import DescriptionRequest, ConceptEdge, ConceptGraphResponse

app = FastAPI(title="Text to Graph API")

@app.post("/generate_graph", response_model=ConceptGraphResponse)
def generate_graph(request: DescriptionRequest):
    text = request.text

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
    concepts_list = df2Graph(df, model="qwen3-coder:480b-cloud")
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