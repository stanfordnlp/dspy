from unittest.mock import patch

import pytest

import dspy
from dspy.teleprompt import MIPROv2
from dspy.utils.dummies import DummyLM


class _SeedCaptured(Exception):
    pass


@pytest.mark.parametrize("compile_seed, expected", [(0, 0), (None, 9), (5, 5)])
def test_compile_seed_overrides_constructor_seed(compile_seed, expected):
    lm = DummyLM([])
    optimizer = MIPROv2(
        metric=lambda example, pred, trace=None: 1.0, prompt_model=lm, task_model=lm, seed=9
    )

    with patch.object(MIPROv2, "_set_random_seeds", side_effect=_SeedCaptured) as set_seeds:
        with pytest.raises(_SeedCaptured):
            optimizer.compile(dspy.Predict("question -> answer"), trainset=[], seed=compile_seed)

    set_seeds.assert_called_once_with(expected)
