from unittest import mock
import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2


def test_mipro_v2_compile_seed_zero():
    """Verify that compile(seed=0) is not treated as falsy and respects seed=0."""
    optimizer = MIPROv2(
        metric=lambda x, y: 1.0,
        auto="light",
        prompt_model=mock.MagicMock(),
        task_model=mock.MagicMock(),
        seed=42,
    )

    with (
        mock.patch.object(optimizer, "_set_random_seeds") as mock_set_seeds,
        mock.patch.object(optimizer, "_set_and_validate_datasets", return_value=([], [])),
        mock.patch.object(optimizer, "_bootstrap_fewshot_examples", return_value={}),
        mock.patch.object(optimizer, "_propose_instructions", return_value={}),
        mock.patch.object(optimizer, "_optimize_prompt_parameters", return_value=mock.MagicMock()),
    ):
        student = mock.MagicMock(spec=dspy.Module)
        optimizer.compile(student, trainset=[], seed=0)
        mock_set_seeds.assert_called_once_with(0)


def test_mipro_v2_compile_seed_fallback_to_init():
    """Verify that compile(seed=None) falls back to self.seed."""
    optimizer = MIPROv2(
        metric=lambda x, y: 1.0,
        auto="light",
        prompt_model=mock.MagicMock(),
        task_model=mock.MagicMock(),
        seed=42,
    )

    with (
        mock.patch.object(optimizer, "_set_random_seeds") as mock_set_seeds,
        mock.patch.object(optimizer, "_set_and_validate_datasets", return_value=([], [])),
        mock.patch.object(optimizer, "_bootstrap_fewshot_examples", return_value={}),
        mock.patch.object(optimizer, "_propose_instructions", return_value={}),
        mock.patch.object(optimizer, "_optimize_prompt_parameters", return_value=mock.MagicMock()),
    ):
        student = mock.MagicMock(spec=dspy.Module)
        optimizer.compile(student, trainset=[], seed=None)
        mock_set_seeds.assert_called_once_with(42)
