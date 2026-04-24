import json
import csv
import requests
from collections import defaultdict

BASE_URL = "http://localhost:8000"

ASK_URL = f"{BASE_URL}/ask"
ASK_WITH_GRAPH_URL = f"{BASE_URL}/ask-with-graph"


# -----------------------------
# Load JSON
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Build graph + prereqs
# -----------------------------
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


# -----------------------------
# Find roots
# -----------------------------
def find_roots(nodes, prereq):
    return [n for n in nodes if len(prereq[n]) == 0]


# -----------------------------
# DFS: extract all branches (root -> leaf paths)
# -----------------------------
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


# -----------------------------
# Knowledge state
# -----------------------------
def build_knowledge(nodes, learned):
    return {n: (100 if n in learned else 0) for n in nodes}


# -----------------------------
# API calls
# -----------------------------
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


# -----------------------------
# utils
# -----------------------------
def norm(x):
    return x.strip().lower()


def is_correct(pred, expected):
    return norm(pred) == norm(expected)


# -----------------------------
# Evaluate one branch
# -----------------------------
def evaluate_branch(branch, nodes, edges, graph, prereq):
    results = []

    learned = set(branch)

    # 🔥 HANDLE EMPTY STATE
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

    results.append({
        "mode": "no_graph",
        "branch": " -> ".join(branch) if branch else "EMPTY",
        "learned": ",".join(sorted(learned)),
        "expected": ",".join(expected_next),
        "actual": r1,
        "is_correct": norm(r1) in {norm(x) for x in expected_next},
    })

    results.append({
        "mode": "with_graph",
        "branch": " -> ".join(branch) if branch else "EMPTY",
        "learned": ",".join(sorted(learned)),
        "expected": ",".join(expected_next),
        "actual": r2,
        "is_correct": norm(r2) in {norm(x) for x in expected_next},
    })

    return results


# -----------------------------
# Run
# -----------------------------
def run():
    concepts_data = load_json("test_data/general/concepts.json")
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
        all_results.extend(evaluate_branch(branch, nodes, edges, graph, prereq))

    return all_results


# -----------------------------
# Save CSV
# -----------------------------
def save(results, filename="branch_evaluation2.csv"):
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
        writer.writerows(results)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    results = run()
    save(results)
    print("Done → branch_evaluation.csv")