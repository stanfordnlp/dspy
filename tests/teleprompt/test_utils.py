from unittest.mock import Mock
import pytest

import dspy
from dspy.teleprompt.utils import eval_candidate_program

class DummyModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        pass


def test_eval_candidate_program_full_trainset():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(return_value=0)
    batch_size = 10

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    evaluate.assert_called_once()
    _, called_kwargs = evaluate.call_args
    assert len(called_kwargs['devset']) == len(trainset)
    assert result == 0

def test_eval_candidate_program_minibatch():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(return_value=0)
    batch_size = 3

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    evaluate.assert_called_once()
    _, called_kwargs = evaluate.call_args
    assert len(called_kwargs['devset']) == batch_size
    assert result == 0

def test_eval_candidate_program_failure():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(side_effect=ValueError("Error"))
    batch_size = 3

    with pytest.raises(ValueError, match="Error"):
        eval_candidate_program(batch_size, trainset, candidate_program, evaluate)