import json
import csv
import requests

BASE_URL = "http://localhost:8000"

ASK_URL = f"{BASE_URL}/ask"
ASK_WITH_GRAPH_URL = f"{BASE_URL}/ask-with-graph"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_knowledge_state(concepts, known_count):
    knowledge = {}
    for i, concept in enumerate(concepts):
        knowledge[concept] = 100 if i < known_count else 0
    return knowledge


def get_expected(concepts, known_count):
    if known_count >= len(concepts):
        return None
    return concepts[known_count]


def normalize(text):
    return text.strip().lower()


def call_ask(knowledge):
    payload = {
        "userPrompt": "What concept should I learn next? Respond with ONLY the concept name.",
        "knowledge": knowledge
    }
    r = requests.post(ASK_URL, json=payload)
    r.raise_for_status()
    return r.json()["response"]


def call_ask_with_graph(knowledge, graph_edges):
    payload = {
        "userPrompt": "What concept should I learn next? Respond with ONLY the concept name.",
        "knowledge": knowledge,
        "graphEdges": graph_edges
    }
    r = requests.post(ASK_WITH_GRAPH_URL, json=payload)
    r.raise_for_status()
    return r.json()["response"]


def is_correct(expected, actual):
    if expected is None:
        # Last concept → no strict expectation
        return True

    return normalize(actual) == normalize(expected)


def run_tests():
    concepts_data = load_json("test_data/linear/concepts.json")
    graph_data = load_json("test_data/linear/graph.json")

    concepts = concepts_data["concepts"]
    graph_edges = graph_data["edges"]

    evaluation_results = []

    for known_count in range(len(concepts) + 1):
        knowledge = build_knowledge_state(concepts, known_count)
        expected = get_expected(concepts, known_count)

        try:
            actual_no_graph = call_ask(knowledge)
        except Exception as e:
            actual_no_graph = f"ERROR: {e}"

        try:
            actual_with_graph = call_ask_with_graph(knowledge, graph_edges)
        except Exception as e:
            actual_with_graph = f"ERROR: {e}"

        evaluation_results.append({
            "mode": "no_graph",
            "known_count": known_count,
            "expected": expected,
            "actual": actual_no_graph,
            "is_correct": is_correct(expected, actual_no_graph)
        })

        evaluation_results.append({
            "mode": "with_graph",
            "known_count": known_count,
            "expected": expected,
            "actual": actual_with_graph,
            "is_correct": is_correct(expected, actual_with_graph)
        })

    return evaluation_results


def save_to_csv(evaluation_results, filename="test_output/linear/resultsIteration3.csv"):
    keys = ["mode", "known_count", "expected", "actual", "is_correct"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(evaluation_results)


if __name__ == "__main__":
    results = run_tests()
    save_to_csv(results)
    print("Tests completed. Results saved to results.csv")