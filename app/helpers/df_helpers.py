import uuid
import pandas as pd
import numpy as np
from .prompts import graph_prompt_dag


def documents_to_dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def dataframe_to_edges(dataframe: pd.DataFrame) -> list:
    results = dataframe.apply(
        lambda row: graph_prompt_dag(row.text, {"chunk_id": row.chunk_id}), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    if len(results) == 1:
        return results[0]

    print(results)
    all_edges = np.concatenate(results).ravel().tolist()
    return all_edges


def edges_to_dataframe(nodes_list) -> pd.DataFrame:
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe
