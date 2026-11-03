import pytest

from dspy.utils.dummies import DummyLM


@pytest.mark.parametrize("answers", ["text", 42, None])
def test_dummy_lm_rejects_answers_that_are_not_list_or_dict(answers):
    with pytest.raises(TypeError, match="must be a list of response dicts or a dict"):
        DummyLM(answers)


def test_dummy_lm_accepts_a_list_and_a_dict():
    assert list(DummyLM([{"answer": "red"}]).answers) == [{"answer": "red"}]
    assert DummyLM({"prompt": {"answer": "red"}}).answers == {"prompt": {"answer": "red"}}
