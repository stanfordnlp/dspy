from unittest.mock import MagicMock, patch

import pytest

import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2, _has_multimodal_inputs


def test_compile_respects_seed_zero():
    """compile(seed=0) must keep seed 0 instead of falling back to the constructor default.

    Regression test for https://github.com/stanfordnlp/dspy/issues/10321
    """
    lm = MagicMock()
    optimizer = MIPROv2(metric=lambda *args, **kwargs: True, auto="light", prompt_model=lm, task_model=lm)

    with (
        patch.object(optimizer, "_set_random_seeds") as mock_set_seeds,
        patch.object(optimizer, "_set_and_validate_datasets", side_effect=RuntimeError("stop after seed")),
    ):
        with pytest.raises(RuntimeError, match="stop after seed"):
            optimizer.compile(MagicMock(), trainset=[], seed=0)

    mock_set_seeds.assert_called_once_with(0)


def test_has_multimodal_inputs():
    """_has_multimodal_inputs detects programs or datasets using Image or Type."""
    text_program = dspy.Predict("question -> answer")
    assert not _has_multimodal_inputs(text_program, trainset=[])

    class VisualQA(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    vqa_program = dspy.Predict(VisualQA)
    assert _has_multimodal_inputs(vqa_program, trainset=[])

    trainset_with_image = [
        dspy.Example(
            image=dspy.Image("data:image/png;base64,AAAA"),
            question="q",
        ).with_inputs("image", "question")
    ]
    assert _has_multimodal_inputs(text_program, trainset=trainset_with_image)

