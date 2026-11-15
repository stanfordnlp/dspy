import pytest

from dspy.dsp.utils import batch, dotdict, print_message


def test_dotdict_attribute_access():
    d = dotdict(a=1, b="test")
    assert d.a == 1
    assert d.b == "test"
    assert d["a"] == 1

    d.c = 3.14
    assert d["c"] == 3.14
    assert d.c == 3.14

    del d.a
    assert "a" not in d
    assert not hasattr(d, "a")


def test_dotdict_dunder_attribute_lookup():
    d = dotdict(a=1)
    # Standard class attribute lookup succeeds
    assert d.__class__ is dotdict

    # Accessing non-existent dunder attribute raises AttributeError cleanly
    with pytest.raises(AttributeError, match="object has no attribute '__missing_dunder__'"):
        _ = d.__missing_dunder__

    # Custom dunder key in dict store is accessible
    d["__custom__"] = 42
    assert d.__custom__ == 42


def test_dotdict_delattr_missing_raises_attribute_error():
    d = dotdict(a=1)
    with pytest.raises(AttributeError, match="object has no attribute 'non_existent'"):
        del d.non_existent


def test_batch_splits_correctly():
    items = list(range(10))
    batches = list(batch(items, bsize=3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    batches_with_offset = list(batch(items, bsize=4, provide_offset=True))
    assert batches_with_offset == [(0, [0, 1, 2, 3]), (4, [4, 5, 6, 7]), (8, [8, 9])]


def test_batch_invalid_bsize_raises_value_error():
    with pytest.raises(ValueError, match="bsize must be a positive integer"):
        list(batch([1, 2, 3], bsize=0))

    with pytest.raises(ValueError, match="bsize must be a positive integer"):
        list(batch([1, 2, 3], bsize=-2))


def test_print_message_sep():
    msg = print_message("one", "two", "three", sep=" | ", condition=False)
    assert "one | two | three" in msg

    msg_default = print_message("one", "two", "three", condition=False)
    assert "one two three" in msg_default
