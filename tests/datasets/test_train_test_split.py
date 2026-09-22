import random

import pytest

from dspy import Example
from dspy.datasets.dataloader import DataLoader


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.mark.parametrize("seed", [0, 42])
def test_seeded_split_preserves_global_random_state(seed):
    examples = [Example(value=i) for i in range(10)]
    random.seed(123)
    state = random.getstate()

    DataLoader().train_test_split(examples, random_state=seed)

    assert random.getstate() == state


@pytest.mark.parametrize("seed", [0, 42])
def test_seeded_split_preserves_deterministic_order_and_input(seed):
    examples = [Example(value=i) for i in range(10)]
    original = examples.copy()
    expected = examples.copy()
    random.Random(seed).shuffle(expected)

    first = DataLoader().train_test_split(examples, train_size=6, test_size=3, random_state=seed)
    random.seed(456)
    second = DataLoader().train_test_split(examples, train_size=6, test_size=3, random_state=seed)

    assert first == second == {"train": expected[:6], "test": expected[6:9]}
    assert all(actual is initial for actual, initial in zip(examples, original, strict=True))


def test_unseeded_split_uses_global_random_state():
    examples = [Example(value=i) for i in range(10)]
    random.seed(123)
    expected = examples.copy()
    random.shuffle(expected)
    expected_state = random.getstate()
    random.seed(123)

    split = DataLoader().train_test_split(examples, train_size=0.6)

    assert split == {"train": expected[:6], "test": expected[6:]}
    assert random.getstate() == expected_state


def test_invalid_seeded_split_preserves_global_random_state():
    examples = [Example(value=i) for i in range(10)]
    state = random.getstate()

    with pytest.raises(ValueError, match="Invalid `train_size`"):
        DataLoader().train_test_split(examples, train_size=1.5, random_state=42)

    assert random.getstate() == state
