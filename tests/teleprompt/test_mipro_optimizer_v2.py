from unittest.mock import MagicMock, patch

import pytest

from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2


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
