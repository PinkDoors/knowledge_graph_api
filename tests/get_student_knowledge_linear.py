import json
import random


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


def get_display_order(concepts, seed):
    rng = random.Random(seed)
    order = concepts[:]
    rng.shuffle(order)
    return order


def run_tests():
    concepts_data = load_json("test_data/linear/concepts.json")
    concepts = concepts_data["concepts"]

    display_order = get_display_order(concepts, 52)

    for known_count in range(len(concepts) + 1):
        knowledge = build_knowledge_state(concepts, known_count)

        # ✅ preserve randomized order but filter only mastered
        mastered_ordered = [c for c in display_order if knowledge[c] == 100]

        expected = get_expected(concepts, known_count)

        print("Mastered:", mastered_ordered)
        # print("Expected next:", expected)
        print("-" * 50)


if __name__ == "__main__":
    run_tests()