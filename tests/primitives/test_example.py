import pytest
from dspy import Example


def test_example_initialization():
    example = Example(a=1, b=2)
    assert example.a == 1
    assert example.b == 2


def test_example_initialization_from_base():
    base = Example(a=1, b=2)
    example = Example(base=base, c=3)
    assert example.a == 1
    assert example.b == 2
    assert example.c == 3


def test_example_initialization_from_dict():
    base_dict = {"a": 1, "b": 2}
    example = Example(base=base_dict, c=3)
    assert example.a == 1
    assert example.b == 2
    assert example.c == 3


def test_example_set_get_item():
    example = Example()
    example["a"] = 1
    assert example["a"] == 1


def test_example_attribute_access():
    example = Example(a=1)
    assert example.a == 1
    example.a = 2
    assert example.a == 2


def test_example_deletion():
    example = Example(a=1, b=2)
    del example["a"]
    with pytest.raises(AttributeError):
        _ = example.a


def test_example_len():
    example = Example(a=1, b=2, dspy_hidden=3)
    assert len(example) == 2


def test_example_repr_str():
    example = Example(a=1)
    assert repr(example) == "Example({'a': 1}) (input_keys=None)"
    assert str(example) == "Example({'a': 1}) (input_keys=None)"


def test_example_eq():
    example1 = Example(a=1, b=2)
    example2 = Example(a=1, b=2)
    assert example1 == example2


def test_example_hash():
    example1 = Example(a=1, b=2)
    example2 = Example(a=1, b=2)
    assert hash(example1) == hash(example2)


def test_example_keys_values_items():
    example = Example(a=1, b=2, dspy_hidden=3)
    assert set(example.keys()) == {"a", "b"}
    assert 1 in example.values()
    assert ("b", 2) in example.items()


def test_example_get():
    example = Example(a=1, b=2)
    assert example.get("a") == 1
    assert example.get("c", "default") == "default"


def test_example_with_inputs():
    example = Example(a=1, b=2).with_inputs("a")
    assert example._input_keys == {"a"}


def test_example_inputs_labels():
    example = Example(a=1, b=2).with_inputs("a")
    inputs = example.inputs()
    assert inputs.toDict() == {"a": 1}
    labels = example.labels()
    assert labels.toDict() == {"b": 2}


def test_example_copy_without():
    example = Example(a=1, b=2)
    copied = example.copy(c=3)
    assert copied.a == 1
    assert copied.c == 3
    without_a = copied.without("a")
    with pytest.raises(AttributeError):
        _ = without_a.a


def test_example_to_dict():
    example = Example(a=1, b=2)
    assert example.toDict() == {"a": 1, "b": 2}
