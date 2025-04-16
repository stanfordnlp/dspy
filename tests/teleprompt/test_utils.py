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
    assert called_kwargs['callback_metadata'] == {"metric_key": "eval_full"}
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
    assert called_kwargs['callback_metadata'] == {"metric_key": "eval_minibatch"}
    assert result == 0

@pytest.mark.parametrize("return_all_scores", [True, False])
def test_eval_candidate_program_failure(return_all_scores):
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(side_effect=ValueError("Error"))
    batch_size = 3

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate, return_all_scores=return_all_scores)

    if return_all_scores:
        assert result == (0.0, [0.0]*len(trainset))
    else:
        assert result == 0.0
