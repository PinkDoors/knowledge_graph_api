def validate_question(q):
    if q["type"] == "multiple_choice_question":
        assert len(q["options"]) == 4
        assert isinstance(q["correct_index"], list)

    elif q["type"] == "true_false_question":
        assert q["options"] == ["True", "False"]
        assert isinstance(q["correct_index"], list)

    elif q["type"] == "multiple_answers_question":
        assert len(q["options"]) >= 3
        assert isinstance(q["correct_index"], list)
        assert len(q["correct_index"]) >= 2