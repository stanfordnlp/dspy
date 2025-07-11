import inspect

import pydantic
import pytest

import dspy


def test_code_validate_input():
    # Create a `dspy.Code` instance with valid code.
    code = dspy.Code(code="print('Hello, world!')")
    assert code.code == "print('Hello, world!')"

    with pytest.raises(ValueError):
        # Try to create a `dspy.Code` instance with invalid type.
        dspy.Code(code=123)

    def foo(x):
        return x + 1

    code_source = inspect.getsource(foo)
    code = dspy.Code(code=code_source)

    assert code.code == code_source


def test_code_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        code: dspy.Code

    code = dspy.Code(code="print('Hello, world!')")
    wrapper = Wrapper(code=code)
    assert wrapper.code.code == "print('Hello, world!')"


def test_code_parses_from_dirty_code():
    dirty_code = "```python\nprint('Hello, world!')```"
    code = dspy.Code(code=dirty_code)
    assert code.code == "print('Hello, world!')"

    dirty_code_with_reasoning = """
The generated code is:
```python
print('Hello, world!')
```

The reasoning is:
The code is a simple print statement.
"""
    code = dspy.Code(code=dirty_code_with_reasoning)
    assert code.code == "print('Hello, world!')"
