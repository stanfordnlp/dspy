"""Test Evaluate with dspy.History objects."""
import json
import tempfile

import dspy
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM


def test_evaluate_save_as_json_with_history():
    """Test that save_as_json works with Examples containing dspy.History objects."""
    # Setup
    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        )
    )

    # Create history objects
    history1 = dspy.History(
        messages=[
            {"question": "Previous Q1", "answer": "Previous A1"},
        ]
    )
    history2 = dspy.History(
        messages=[
            {"question": "Previous Q2", "answer": "Previous A2"},
            {"question": "Previous Q3", "answer": "Previous A3"},
        ]
    )

    # Create examples with history
    devset = [
        dspy.Example(question="What is 1+1?", answer="2", history=history1).with_inputs("question"),
        dspy.Example(question="What is 2+2?", answer="4", history=history2).with_inputs("question"),
    ]

    program = Predict("question -> answer")

    # Create evaluator with save_as_json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_json = f.name

    try:
        evaluator = Evaluate(
            devset=devset,
            metric=answer_exact_match,
            display_progress=False,
            save_as_json=temp_json,
        )

        result = evaluator(program)
        assert result.score == 100.0

        # Verify JSON file was created and is valid
        with open(temp_json) as f:
            data = json.load(f)

        assert len(data) == 2

        # Verify history was properly serialized in first record
        assert "history" in data[0]
        assert isinstance(data[0]["history"], dict)
        assert "messages" in data[0]["history"]
        assert len(data[0]["history"]["messages"]) == 1
        assert data[0]["history"]["messages"][0] == {"question": "Previous Q1", "answer": "Previous A1"}

        # Verify history was properly serialized in second record
        assert "history" in data[1]
        assert isinstance(data[1]["history"], dict)
        assert "messages" in data[1]["history"]
        assert len(data[1]["history"]["messages"]) == 2
        assert data[1]["history"]["messages"][0] == {"question": "Previous Q2", "answer": "Previous A2"}
        assert data[1]["history"]["messages"][1] == {"question": "Previous Q3", "answer": "Previous A3"}

    finally:
        import os
        if os.path.exists(temp_json):
            os.unlink(temp_json)


def test_evaluate_save_as_csv_with_history():
    """Test that save_as_csv works with Examples containing dspy.History objects."""
    # Setup
    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
            }
        )
    )

    # Create history object
    history = dspy.History(
        messages=[
            {"question": "Previous Q", "answer": "Previous A"},
        ]
    )

    # Create example with history
    devset = [
        dspy.Example(question="What is 1+1?", answer="2", history=history).with_inputs("question"),
    ]

    program = Predict("question -> answer")

    # Create evaluator with save_as_csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_csv = f.name

    try:
        evaluator = Evaluate(
            devset=devset,
            metric=answer_exact_match,
            display_progress=False,
            save_as_csv=temp_csv,
        )

        result = evaluator(program)
        assert result.score == 100.0

        # Verify CSV file was created
        import csv
        with open(temp_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert "history" in rows[0]
        # CSV will have string representation of the dict
        assert "messages" in rows[0]["history"]

    finally:
        import os
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)
