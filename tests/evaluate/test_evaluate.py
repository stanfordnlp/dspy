import json
import signal
import tempfile
import threading
from unittest.mock import patch

import pytest

import dspy
from dspy.evaluate.evaluate import Evaluate, EvaluationResult
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import Predict
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


def new_example(question, answer):
    """Helper function to create a new example."""
    return dspy.Example(
        question=question,
        answer=answer,
    ).with_inputs("question")


def test_evaluate_initialization():
    devset = [new_example("What is 1+1?", "2")]
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    assert ev.devset == devset
    assert ev.metric == answer_exact_match
    assert ev.num_threads is None
    assert not ev.display_progress


def test_evaluate_call():
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        )
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    assert program(question="What is 1+1?").answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = ev(program)
    assert score.score == 100.0


def test_evaluate_single_thread_runs_in_main_thread():
    """Evaluate with num_threads=1 should run in the main thread."""
    dspy.configure(
        lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}})
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]

    execution_threads = []

    original_metric = answer_exact_match

    def tracking_metric(example, prediction, **kwargs):
        execution_threads.append(threading.current_thread())
        return original_metric(example, prediction, **kwargs)

    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=tracking_metric,
        display_progress=False,
        num_threads=1,
    )
    result = ev(program)
    assert result.score == 100.0
    assert all(t is threading.main_thread() for t in execution_threads)


@pytest.mark.extra
def test_construct_result_df():
    import pandas as pd
    devset = [
        new_example("What is 1+1?", "2"),
        new_example("What is 2+2?", "4"),
        new_example("What is 3+3?", "-1"),
    ]
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
    )
    results = [
        (devset[0], {"answer": "2"}, 100.0),
        (devset[1], {"answer": "4"}, 100.0),
        (devset[2], {"answer": "-1"}, 0.0),
    ]
    result_df = ev._construct_result_table(results, answer_exact_match.__name__)
    pd.testing.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "question": ["What is 1+1?", "What is 2+2?", "What is 3+3?"],
                "example_answer": ["2", "4", "-1"],
                "pred_answer": ["2", "4", "-1"],
                "answer_exact_match": [100.0, 100.0, 0.0],
            }
        ),
    )


def test_multithread_evaluate_call():
    dspy.configure(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    assert program(question="What is 1+1?").answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
        num_threads=2,
    )
    result = ev(program)
    assert result.score == 100.0


def test_multi_thread_evaluate_call_cancelled(monkeypatch):
    # slow LM that sleeps for 1 second before returning the answer
    class SlowLM(DummyLM):
        def __call__(self, *args, **kwargs):
            import time

            time.sleep(1)
            return super().__call__(*args, **kwargs)

    dspy.configure(lm=SlowLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))

    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    assert program(question="What is 1+1?").answer == "2"

    # spawn a thread that will sleep for .1 seconds then send a KeyboardInterrupt
    def sleep_then_interrupt():
        import time

        time.sleep(0.1)
        import os

        os.kill(os.getpid(), signal.SIGINT)

    input_thread = threading.Thread(target=sleep_then_interrupt)
    input_thread.start()

    with pytest.raises(KeyboardInterrupt):
        ev = Evaluate(
            devset=devset,
            metric=answer_exact_match,
            display_progress=False,
            num_threads=2,
        )
        ev(program)


def test_evaluate_call_wrong_answer():
    dspy.configure(lm=DummyLM({"What is 1+1?": {"answer": "0"}, "What is 2+2?": {"answer": "0"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    result = ev(program)
    assert result.score == 0.0


@pytest.mark.extra
@pytest.mark.parametrize(
    "program_with_example",
    [
        (Predict("question -> answer"), new_example("What is 1+1?", "2")),
        # Create programs that do not return dictionary-like objects because Evaluate()
        # has failed for such cases in the past
        (
            lambda text: Predict("text: str -> entities: list[str]")(text=text).entities,
            dspy.Example(text="United States", entities=["United States"]).with_inputs("text"),
        ),
        (
            lambda text: Predict("text: str -> entities: list[dict[str, str]]")(text=text).entities,
            dspy.Example(text="United States", entities=[{"name": "United States", "type": "location"}]).with_inputs(
                "text"
            ),
        ),
        (
            lambda text: Predict("text: str -> first_word: Tuple[str, int]")(text=text).words,
            dspy.Example(text="United States", first_word=("United", 6)).with_inputs("text"),
        ),
    ],
)
@pytest.mark.parametrize("display_table", [True, False, 1])
@pytest.mark.parametrize("is_in_ipython_notebook_environment", [True, False])
def test_evaluate_display_table(program_with_example, display_table, is_in_ipython_notebook_environment, capfd):
    program, example = program_with_example
    example_input = next(iter(example.inputs().values()))
    example_output = {key: value for key, value in example.toDict().items() if key not in example.inputs()}

    dspy.configure(
        lm=DummyLM(
            {
                example_input: example_output,
            }
        )
    )

    ev = Evaluate(
        devset=[example],
        metric=lambda example, pred, **kwargs: example == pred,
        display_table=display_table,
    )
    assert ev.display_table == display_table

    with patch(
        "dspy.evaluate.evaluate.is_in_ipython_notebook_environment", return_value=is_in_ipython_notebook_environment
    ):
        ev(program)
        out, _ = capfd.readouterr()
        if not is_in_ipython_notebook_environment and display_table:
            # In console environments where IPython is not available, the table should be printed
            # to the console
            example_input = next(iter(example.inputs().values()))
            assert example_input in out


def test_evaluate_callback():
    class TestCallback(BaseCallback):
        def __init__(self):
            self.start_call_inputs = None
            self.start_call_count = 0
            self.end_call_outputs = None
            self.end_call_count = 0

        def on_evaluate_start(
            self,
            call_id: str,
            instance,
            inputs,
        ):
            self.start_call_inputs = inputs
            self.start_call_count += 1

        def on_evaluate_end(
            self,
            call_id: str,
            outputs,
            exception=None,
        ):
            self.end_call_outputs = outputs
            self.end_call_count += 1

    callback = TestCallback()
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        ),
        callbacks=[callback],
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    assert program(question="What is 1+1?").answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    result = ev(program)
    assert result.score == 100.0
    assert callback.start_call_inputs["program"] == program
    assert callback.start_call_count == 1
    assert callback.end_call_outputs.score == 100.0
    assert callback.end_call_count == 1

def test_evaluation_result_repr():
    result = EvaluationResult(score=100.0, results=[(new_example("What is 1+1?", "2"), {"answer": "2"}, 100.0)])
    assert repr(result) == "EvaluationResult(score=100.0, results=<list of 1 results>)"


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


def test_evaluate_population_metric_only():
    """Test Evaluate with only a population_metric (no per-sample metric)."""
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
                "What is 3+3?": {"answer": "6"},
            }
        )
    )
    devset = [
        new_example("What is 1+1?", "2"),
        new_example("What is 2+2?", "4"),
        new_example("What is 3+3?", "6"),
    ]
    program = Predict("question -> answer")

    def pop_metric(examples, predictions):
        # Simple population metric: fraction of exact matches
        correct = sum(
            1 for ex, pred in zip(examples, predictions) if ex.answer == pred.answer
        )
        return round(correct / len(examples), 2)

    ev = Evaluate(
        devset=devset,
        population_metric=pop_metric,
        display_progress=False,
    )
    result = ev(program)
    assert result.score == 1.0
    assert len(result.results) == 3
    # Per-sample scores should be None when no sample-wise metric is given
    for _, _, score in result.results:
        assert score is None


def test_evaluate_population_metric_with_sample_metric():
    """Test Evaluate with both a per-sample metric and a population_metric.
    The per-sample metric should still produce per-row scores, but the
    overall EvaluationResult.score should come from population_metric.
    """
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        )
    )
    devset = [
        new_example("What is 1+1?", "2"),
        new_example("What is 2+2?", "4"),
    ]
    program = Predict("question -> answer")

    def pop_metric(examples, predictions):
        # Always returns a fixed value so we can assert it overrides the sample average
        return 42.0

    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        population_metric=pop_metric,
        display_progress=False,
    )
    result = ev(program)
    # The overall score comes from the population_metric, not the per-sample average
    assert result.score == 42.0
    # But per-sample scores should still be present from the sample metric
    for _, _, score in result.results:
        assert score is not None


def test_evaluate_population_metric_pearson():
    """End-to-end test using Pearson correlation as a population_metric.
    This is the motivating use-case from issue #9076.
    """
    from scipy.stats import pearsonr

    dspy.configure(
        lm=DummyLM(
            {
                "Rate item A": {"rating": "5"},
                "Rate item B": {"rating": "3"},
                "Rate item C": {"rating": "1"},
                "Rate item D": {"rating": "4"},
            }
        )
    )
    devset = [
        dspy.Example(text="Rate item A", rating=5).with_inputs("text"),
        dspy.Example(text="Rate item B", rating=3).with_inputs("text"),
        dspy.Example(text="Rate item C", rating=1).with_inputs("text"),
        dspy.Example(text="Rate item D", rating=4).with_inputs("text"),
    ]
    program = Predict("text -> rating")

    def pearson_population_metric(examples, predictions):
        ground_truth = [float(ex.rating) for ex in examples]
        predicted = [float(pred.rating) for pred in predictions]
        r, _ = pearsonr(ground_truth, predicted)
        return round(r, 4)

    ev = Evaluate(
        devset=devset,
        population_metric=pearson_population_metric,
        display_progress=False,
    )
    result = ev(program)
    # The DummyLM returns perfect predictions, so Pearson r should be 1.0
    assert result.score == 1.0


def test_evaluate_backward_compatible_without_population_metric():
    """Ensure that existing behavior is unchanged when population_metric is not provided."""
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        )
    )
    devset = [
        new_example("What is 1+1?", "2"),
        new_example("What is 2+2?", "4"),
    ]
    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    result = ev(program)
    # Original behavior: score is the percentage of correct answers
    assert result.score == 100.0
    assert len(result.results) == 2

