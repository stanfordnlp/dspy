import dsp
from dspy import ChainOfThoughtWithHint
from dspy.utils import DummyLM

def test_cot_with_no_hint():
    dsp.settings.lm = DummyLM(["find the number after 1", "2"])
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == ["rationale", "hint", "answer"]
    assert predict(question="What is 1+1?").answer == "2"

    final_convo = dsp.settings.lm.get_convo(-1)
    assert final_convo.endswith(
        "Question: What is 1+1?\n"
        "Reasoning: Let's think step by step in order to find the number after 1\n"
        "Answer: 2")

def test_cot_with_hint():
    dsp.settings.lm = DummyLM(["find the number after 1", "2"])
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == ["rationale", "hint", "answer"]
    assert predict(question="What is 1+1?", hint="think small").answer == "2"

    final_convo = dsp.settings.lm.get_convo(-1)
    assert final_convo.endswith(
        "Question: What is 1+1?\n\n"
        "Reasoning: Let's think step by step in order to find the number after 1\n\n"
        "Hint: think small\n\n"
        "Answer: 2")