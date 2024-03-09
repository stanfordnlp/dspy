import dsp, dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
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
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": "2", "What is 2+2?": "4"}))
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

def test_evaluate_call_bad():
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": "0", "What is 2+2?": "0"}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = ev(program)
    assert score == 0.0

def test_evaluate_display_table():
    devset = [new_example("What is 1+1?", "2")]
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_table=True,
    )
    assert ev.display_table == True

