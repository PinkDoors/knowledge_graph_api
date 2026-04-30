import sys

sys.path.append("..")

import json
from app.ollama import client

model = "gpt-oss:120b-cloud"

def graph_prompt_linear(user_input: str, metadata=None):
    if metadata is None:
        metadata = {}
    sys_prompt = (
        '''
You are given a subject description. Your task is to generate a single linear step-by-step learning path for this subject.
Subject description is delimited by ``` and provided to you.

Rules:
The learning path MUST be strictly linear.
If two concepts are learnt simultaneously the order of them is not significant. 
Do NOT branch, skip, or create parallel paths.
Output only the sequence of relationships; no extra explanations.
Each concept MUST be atomic. Do not use "()" and "," symbols. Do not create aggregated concepts that group multiple topics (e.g., ‘basics’, ‘fundamentals’, ‘core concepts’).

OUTPUT FORMAT:
Format your output strictly as a JSON list of ordered relationships. Each element represents one step in the learning path.
Example output (Do not include any other annotations, start with [ symbol immediately):

[
{
node_1: Earlier concept in the learning path,
node_2: Next atomic concept derived from the context,
edge: Explanation of how node_2 builds on node_1
},
{ }
]
        '''
    )

    user_prompt = f"context: ```{user_input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=sys_prompt, prompt=user_prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graph_prompt_dag(user_input: str, metadata=None):
    if metadata is None:
        metadata = {}
    sys_prompt = (
        '''
You are given a subject description. Your task is to generate a SINGLE fully-connected learning graph.

Rules:
The structure MUST be a directed acyclic graph (DAG).
Concepts can branch into multiple sub-concepts.
A concept can depend on multiple prerequisite concepts.
Each concept MUST be atomic. Do not use "()" and "," symbols. Do not create aggregated concepts that group multiple topics (e.g., ‘basics’, ‘fundamentals’, ‘core concepts’).

Output only relationships; no explanations outside edges.

OUTPUT FORMAT:
Return a JSON list of edges. Each edge represents a dependency between two concepts.
Example output (Do not include any other annotations, start with [ symbol immediately):

[
{
node_1: prerequisite concept,
node_2: dependent concept,
edge: Explanation of how node_2 builds on node_1
},
{ }
]
        '''
    )

    user_prompt = f"context: ```{user_input}``` \n\n output: "

    response, _ = client.generate(
        model_name=model,
        system=sys_prompt,
        prompt=user_prompt
    )

    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None

    return result


def quiz_prompt(graph, num_questions=5):
    sys_prompt = """
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

    user_prompt = f"""
Knowledge graph:
{json.dumps(graph, indent=2)}

For each concept generate {num_questions} questions.
"""

    response, _ = client.generate(
        model_name=model,
        system=sys_prompt,
        prompt=user_prompt
    )

    try:
        return json.loads(response)
    except:
        print("\n\nERROR ### Bad LLM response:", response, "\n\n")
        return None


def tutor_prompt(user_input: str, knowledge: list):
    sys_prompt = """
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
2. Suggest which concept to learn next
3. Adapt to mastery levels

--------------------------------
LEARNING STRATEGY
--------------------------------
When suggesting what to study next:

- First, identify concepts with low mastery that serve as foundations for other, more advanced concepts in the graph.
  Prioritize strengthening these foundational weak concepts before moving forward.

- If no weak foundational concepts exist, identify the most logical next learning step.

- Always explain your recommendation.

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

    user_prompt = f"""
Student knowledge:
{json.dumps(knowledge, indent=2)}

User question:
{user_input}

Answer:
"""

    response_text, _ = client.generate(
        model_name=model,
        system=sys_prompt,
        prompt=user_prompt
    )

    return response_text.strip()


def tutor_prompt_with_graph(user_input: str, knowledge: list, graph: list):
    sys_prompt = """
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
2. Suggest which concept to learn next
3. Adapt to mastery levels

--------------------------------
LEARNING STRATEGY
--------------------------------
When suggesting what to study next:

- First, identify concepts with low mastery that serve as foundations for other, more advanced concepts in the graph.
  Prioritize strengthening these foundational weak concepts before moving forward.

- If no weak foundational concepts exist, identify the most logical next learning step.

- Always explain your recommendation.

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

    user_prompt = f"""
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
        system=sys_prompt,
        prompt=user_prompt
    )

    return response_text.strip()


# same as tutor_prompt but give only one concept as a response
def tutor_prompt_v2(user_input: str, knowledge: list):
    sys_prompt = """
You are an intelligent tutoring assistant.

You are given:
1. Student knowledge with mastery levels (0-100)

- 80-100 = strong mastery
- 50-79 = moderate understanding
- 0-49 = weak understanding

--------------------------------
YOUR GOALS
--------------------------------
1. Suggest which concept to learn next

--------------------------------
LEARNING STRATEGY
--------------------------------
When suggesting what to study next:

- First, identify concepts with low mastery that serve as foundations for other, more advanced concepts in the graph.
  Prioritize strengthening these foundational weak concepts before moving forward.

- If no weak foundational concepts exist, identify the most logical next learning step.

--------------------------------
OUTPUT
--------------------------------
Return only concept name
"""

    user_prompt = f"""
Student knowledge:
{json.dumps(knowledge, indent=2)}

User question:
{user_input}

Answer:
"""

    response_text, _ = client.generate(
        model_name=model,
        system=sys_prompt,
        prompt=user_prompt
    )

    return response_text.strip()


# same as tutor_prompt_with_graph but give only one concept as a response
def tutor_prompt_with_graph_v2(user_input: str, knowledge: list, graph: list):
    sys_prompt = """
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
1. Suggest which concept to learn next

--------------------------------
LEARNING STRATEGY
--------------------------------
When suggesting what to study next:

- First, identify concepts with low mastery that serve as foundations for other, more advanced concepts in the graph.
  Prioritize strengthening these foundational weak concepts before moving forward.

- If no weak foundational concepts exist, identify the most logical next learning step.

--------------------------------
OUTPUT
--------------------------------
Return only concept name
"""

    user_prompt = f"""
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
        system=sys_prompt,
        prompt=user_prompt
    )

    return response_text.strip()
