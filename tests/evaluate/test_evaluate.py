import signal
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
    dspy.settings.configure(
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
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
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

    dspy.settings.configure(lm=SlowLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))

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
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": {"answer": "0"}, "What is 2+2?": {"answer": "0"}}))
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

    dspy.settings.configure(
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
    dspy.settings.configure(
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
