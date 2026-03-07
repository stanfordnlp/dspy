import inspect

import pydantic
import pytest

import dspy


def test_code_validate_input():
    # Create a `dspy.Code` instance with valid code.
    code = dspy.Code["python"](code="print('Hello, world!')")
    assert code.code == "print('Hello, world!')"

    with pytest.raises(ValueError):
        # Try to create a `dspy.Code` instance with invalid type.
        dspy.Code["python"](code=123)

    def foo(x):
        return x + 1

    code_source = inspect.getsource(foo)
    code = dspy.Code["python"](code=code_source)

    assert code.code == code_source


def test_code_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        code: dspy.Code

    code = dspy.Code(code="print('Hello, world!')")
    wrapper = Wrapper(code=code)
    assert wrapper.code.code == "print('Hello, world!')"


def test_code_with_language():
    java_code = dspy.Code["java"](code="System.out.println('Hello, world!');")
    assert java_code.code == "System.out.println('Hello, world!');"
    assert java_code.language == "java"
    assert "Programming language: java" in java_code.description()

    cpp_code = dspy.Code["cpp"](code="std::cout << 'Hello, world!' << std::endl;")
    assert cpp_code.code == "std::cout << 'Hello, world!' << std::endl;"
    assert cpp_code.language == "cpp"
    assert "Programming language: cpp" in cpp_code.description()


def test_code_no_default_language():
    """Bare dspy.Code should not default to any language (#9251)."""
    assert dspy.Code.language == ""
    desc = dspy.Code.description()
    assert "Programming language" not in desc
    assert "```\n{code}\n```" in desc


def test_code_json_schema_excludes_docstring():
    """JSON schema should not contain the full class docstring (#9251)."""
    schema = pydantic.TypeAdapter(dspy.Code).json_schema()
    assert "Example 1" not in schema.get("description", "")
    assert "Example 2" not in schema.get("description", "")
    assert "code generation and code analysis" not in schema.get("description", "")

    # Parameterized Code should also have a clean schema
    schema_py = pydantic.TypeAdapter(dspy.Code["python"]).json_schema()
    assert "Example 1" not in schema_py.get("description", "")


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
