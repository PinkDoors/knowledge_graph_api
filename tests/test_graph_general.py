import json
import csv
import requests
from collections import defaultdict

BASE_URL = "http://localhost:8000"

ASK_URL = f"{BASE_URL}/ask"
ASK_WITH_GRAPH_URL = f"{BASE_URL}/ask-with-graph"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_graph(edges):
    graph = defaultdict(list)
    prereq = defaultdict(set)
    nodes = set()

    for e in edges:
        a, b = e["node_1"], e["node_2"]
        graph[a].append(b)
        prereq[b].add(a)
        nodes.add(a)
        nodes.add(b)

    return graph, prereq, nodes

def find_roots(nodes, prereq):
    return [n for n in nodes if len(prereq[n]) == 0]

def dfs(node, graph, path, all_paths):
    all_paths.append(path.copy())
    path.append(node)

    if node not in graph or len(graph[node]) == 0:
        all_paths.append(path.copy())
        path.pop()
        return

    for nxt in graph[node]:
        dfs(nxt, graph, path, all_paths)

    path.pop()


def extract_branches(graph, nodes, prereq):
    roots = find_roots(nodes, prereq)
    all_paths = []

    for r in roots:
        dfs(r, graph, [], all_paths)

    return all_paths

def build_knowledge(nodes, learned):
    return {n: (100 if n in learned else 0) for n in nodes}

def call_no_graph(knowledge):
    r = requests.post(ASK_URL, json={
        "userPrompt": "Next concept ONLY",
        "knowledge": knowledge
    })
    r.raise_for_status()
    return r.json()["response"]


def call_with_graph(knowledge, edges):
    r = requests.post(ASK_WITH_GRAPH_URL, json={
        "userPrompt": "Next concept ONLY",
        "knowledge": knowledge,
        "graphEdges": edges
    })
    r.raise_for_status()
    return r.json()["response"]

def norm(x):
    return x.strip().lower()


def is_correct(pred, expected):
    return norm(pred) == norm(expected)

def evaluate_branch(branch, nodes, edges, prereq):
    evaluation_results = []

    learned = set(branch)

    expected_next = [
    n for n in nodes
    if n not in learned and all(p in learned for p in prereq[n])
    ]

    knowledge = build_knowledge(nodes, learned)

    try:
        r1 = call_no_graph(knowledge)
    except Exception as e:
        r1 = f"ERROR: {e}"

    try:
        r2 = call_with_graph(knowledge, edges)
    except Exception as e:
        r2 = f"ERROR: {e}"

    evaluation_results.append({
        "mode": "no_graph",
        "branch": " -> ".join(branch) if branch else "EMPTY",
        "learned": ",".join(sorted(learned)),
        "expected": ",".join(expected_next),
        "actual": r1,
        "is_correct": norm(r1) in {norm(x) for x in expected_next},
    })

    evaluation_results.append({
        "mode": "with_graph",
        "branch": " -> ".join(branch) if branch else "EMPTY",
        "learned": ",".join(sorted(learned)),
        "expected": ",".join(expected_next),
        "actual": r2,
        "is_correct": norm(r2) in {norm(x) for x in expected_next},
    })

    return evaluation_results

def run():
    graph_data = load_json("test_data/general/graph.json")

    edges = graph_data["edges"]

    graph, prereq, nodes = build_graph(edges)

    branches = extract_branches(graph, nodes, prereq)
    seen = set()
    unique_branches = []

    for b in branches:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_branches.append(b)

    branches = unique_branches

    all_results = []

    for branch in branches:
        all_results.extend(evaluate_branch(branch, nodes, edges, prereq))

    return all_results

def save(evaluation_results, filename="test_output/general/branch_evaluationIteration7.csv"):
    keys = [
        "mode",
        "branch",
        "learned",
        "expected",
        "actual",
        "is_correct"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(evaluation_results)


if __name__ == "__main__":
    results = run()
    save(results)
    print("Done → branch_evaluation.csv")