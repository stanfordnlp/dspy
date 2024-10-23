import signal
import threading
from unittest.mock import patch

import pytest

import dsp
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.functional import TypedPredictor
from dspy.predict import Predict
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
    assert ev.num_threads == len(devset)
    assert ev.display_progress == False


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
    assert score == 100.0


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
    score = ev(program)
    assert score == 100.0


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
        score = ev(program)
        assert score == 100.0


def test_evaluate_call_bad():
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": {"answer": "0"}, "What is 2+2?": {"answer": "0"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = ev(program)
    assert score == 0.0


@pytest.mark.parametrize(
    "program_with_example",
    [
        (
            # Create a program that extracts entities from text and returns them as a list,
            # rather than returning a Predictor() wrapper. This is done intentionally to test
            # the case where the program does not output a dictionary-like object because
            # Evaluate() has failed for this case in the past
            lambda text: TypedPredictor("text: str -> entities: List[str]")(text=text).entities,
            dspy.Example(text="United States", entities=["United States"]).with_inputs("text"),
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
