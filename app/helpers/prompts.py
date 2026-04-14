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

def quizPrompt(graph, num_questions=5, model="qwen3.5:397b-cloud"):
    import json

    SYS_PROMPT = """
You are an educational AI that generates quiz questions strictly based on a provided knowledge graph.

Your goal is to assess a student's understanding of the *content of concepts*, NOT the structure of the graph.

--------------------------------
ALLOWED QUESTION TYPES
--------------------------------
1. multiple_choice_question
   - Exactly 4 options
   - Exactly 1 correct answer

2. true_false_question
   - Options MUST be exactly: ["True", "False"]
   - Exactly 1 correct answer

3. multiple_answers_question
   - Exactly 4 options
   - At least 2 correct answers

--------------------------------
STRICT CONTENT RULES
--------------------------------
- ONLY use information explicitly contained within the provided graph.
- DO NOT introduce external knowledge, assumptions, or examples not grounded in the graph.
- DO NOT mention or test:
  - relationships between nodes
  - graph structure (edges, neighbors, connections, hierarchy)
  - phrases like "according to the graph", "linked concept", etc.
- Each question must test factual or conceptual understanding of a SINGLE concept.

--------------------------------
CONCEPT USAGE RULES
--------------------------------
- Each question MUST include a "concept" field.
- The "concept" MUST:
  - exactly match a node name from the graph
  - be the primary subject of the question
- Do NOT combine multiple concepts in one question.

--------------------------------
QUESTION QUALITY RULES
--------------------------------
- Questions must be clear, specific, and unambiguous.
- Avoid trivial wording (e.g., "What is X?" if more meaningful phrasing is possible).
- Distractors (wrong answers) must be plausible.
- Avoid repeating the same question pattern.
- Make sure that each allowed question typed was used for generation
- Choose different correct indexes

--------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------
- Output ONLY valid JSON (no explanations, no markdown, no comments).
- The output must be a JSON array of objects.
- Each object must follow one of the formats below:

[
  {
    "concept": "string",
    "question": "string",
    "type": "multiple_choice_question",
    "options": ["A", "B", "C", "D"],
    "correct_index": [0]
  },
  {
    "concept": "string",
    "question": "string",
    "type": "true_false_question",
    "options": ["True", "False"],
    "correct_index": [1]
  },
  {
    "concept": "string",
    "question": "string",
    "type": "multiple_answers_question",
    "options": ["A", "B", "C", "D"],
    "correct_index": [0, 2]
  }
]

--------------------------------
FINAL VALIDATION BEFORE OUTPUT
--------------------------------
Before producing the final answer, ensure:
- No question refers to graph structure or relationships
- Every concept exists in the graph
- Each question tests only one concept
- JSON is valid and strictly follows schema
"""

    USER_PROMPT = f"""
Knowledge graph:
{json.dumps(graph, indent=2)}

For each concept generate {num_questions} questions.
"""

    response, _ = client.generate(
        model_name=model,
        system=SYS_PROMPT,
        prompt=USER_PROMPT
    )

    try:
        return json.loads(response)
    except:
        print("\n\nERROR ### Bad LLM response:", response, "\n\n")
        return None

def tutorPrompt(user_input: str, knowledge: list, model="qwen3.5:397b-cloud"):
    """
    Generates an LLM response based on student's knowledge and user input.

    Args:
        user_input (str): The question or instruction from the user.
        knowledge (list): List of dicts [{"concept": str, "mastery": int}]
        model (str): The LLM model to use.

    Returns:
        str: LLM-generated response
    """

    # 1️⃣ System prompt: mastery-aware tutoring instructions
    SYS_PROMPT = """
You are an intelligent tutoring assistant.

You are given:
1. Student knowledge with mastery levels (0-100)

- 80-100 = strong mastery
- 50-79 = moderate understanding
- 0-49 = weak understanding

--------------------------------
YOUR GOALS
--------------------------------
1. Answer user questions clearly
2. Suggest what to learn next when appropriate
3. Adapt to mastery levels

--------------------------------
LEARNING STRATEGY
--------------------------------
When suggesting what to study next:

- First, identify concepts with low mastery that serve as foundations for other, more advanced concepts in the graph.
  Prioritize strengthening these foundational weak concepts before moving forward.

- If no weak foundational concepts exist, identify the most logical next learning step.

- Always explain your recommendation using:
  - the student’s mastery levels
  - logical progression of the subject matter

--------------------------------
STYLE
--------------------------------
- Clear and structured
- Use bullet points when useful
- Avoid long paragraphs

--------------------------------
OUTPUT
--------------------------------
Return plain text only
"""

    # 2️⃣ User prompt: context + question
    USER_PROMPT = f"""
Student knowledge:
{json.dumps(knowledge, indent=2)}

User question:
{user_input}

Answer:
"""

    # 3️⃣ Generate response using Ollama client
    response_text, _ = client.generate(
        model_name=model,
        system=SYS_PROMPT,
        prompt=USER_PROMPT
    )

    return response_text.strip()

def tutorPromptWithGraph(user_input: str, knowledge: list, graph: list, model="qwen3.5:397b-cloud"):

    SYS_PROMPT = """
You are an intelligent tutoring assistant.

You are given:
1. Student knowledge with mastery levels (0-100)
2. A knowledge graph describing relationships between concepts

- 80-100 = strong mastery
- 50-79 = moderate understanding
- 0-49 = weak understanding

--------------------------------
GRAPH USAGE (IMPORTANT)
--------------------------------
Knowledge graph has:
- node_1 and node_2 - concepts
- edge - natural-language explanation of how node_2 builds on node_1 for a learner

You MUST use the graph to:
- Understand how concepts depend on or extend each other
- Identify learning paths between simpler and more advanced concepts
- Follow connections from node_1 → node_2 when reasoning
- Use edge descriptions to explain WHY a concept leads to another

When a concept is mentioned:
- Check connected concepts in the graph
- Use edge descriptions to understand how knowledge flows
- If prerequisites are weak, highlight them before explaining advanced concepts

--------------------------------
YOUR GOALS
--------------------------------
1. Answer user questions clearly
2. Suggest what to learn next when appropriate
3. Adapt to mastery levels
4. Use graph structure to improve reasoning and learning paths

--------------------------------
LEARNING STRATEGY
--------------------------------
When suggesting what to study next:

- First, identify concepts with low mastery that serve as foundations for other, more advanced concepts in the graph.
  Prioritize strengthening these foundational weak concepts before moving forward.

- If no weak foundational concepts exist, continue by following the graph structure to identify the most logical next learning step.

- Always explain your recommendation using:
  - the student’s mastery levels
  - the relationships described in the graph edges

--------------------------------
STYLE
--------------------------------
- Clear and structured
- Use bullet points when useful
- Avoid long paragraphs

--------------------------------
OUTPUT
--------------------------------
Return plain text only
"""

    USER_PROMPT = f"""
Student knowledge:
{json.dumps(knowledge, indent=2)}

Knowledge graph:
{json.dumps(graph, indent=2)}

User question:
{user_input}

Answer:
"""

    response_text, _ = client.generate(
        model_name=model,
        system=SYS_PROMPT,
        prompt=USER_PROMPT
    )

    return response_text.strip()