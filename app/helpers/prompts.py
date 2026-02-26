import sys
from yachalk import chalk
sys.path.append("..")

import json
from app.ollama import client

def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    # client.BASE_URL = "http://host.docker.internal:11434"
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        '''
You are given a subject description. Your task is to generate a single linear step-by-step learning path for this subject.
Subject description is delimited by ``` and provided to you.

Rules:
The learning path MUST be strictly linear.
If two concepts are learnt simultaneously the order of them is not significant. 
Do NOT branch, skip, or create parallel paths.
Output only the sequence of relationships; no extra explanations.
Each concept MUST be atomic. Do not use "()" and "," symbols. 

OUTPUT FORMAT:
Format your output strictly as a JSON list of ordered relationships. Each element represents one step in the learning path.
Example output (Do not include any other annotations, start with [ symbol immediately):

[
{
node_1: Earlier concept in the learning path,
node_2: Next atomic concept derived from the context,
edge: Explanation of how node_2 builds on node_1 for a learner
},
{ }
]
        '''
    )




    # client.BASE_URL = "http://host.docker.internal:11434"
    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result
