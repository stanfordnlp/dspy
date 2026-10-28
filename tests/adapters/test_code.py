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


def test_nested_code_annotation_does_not_leak_docstring_into_prompt():
    """Regression test for #9251: `dspy.Code`'s class docstring (which contains full code
    examples) must not leak into the prompt via the pydantic JSON schema when the type
    appears inside a composite annotation such as `list[dspy.Code]`."""

    class CodeSnippets(dspy.Signature):
        question: str = dspy.InputField()
        snippets: list[dspy.Code] = dspy.OutputField()

    messages = dspy.ChatAdapter().format(CodeSnippets, [], {"question": "reverse a string"})
    system_content = messages[0]["content"]

    # Docstring content (example code) must not appear in the prompt.
    assert "sleepsort" not in system_content
    assert "gpt-4o-mini" not in system_content
    # The structural JSON schema for the list is still communicated.
    assert "adhere to the JSON schema" in system_content
    # The intended, LLM-facing type description is still present.
    assert "Type description of Code" in system_content


def test_explicit_json_schema_description_is_preserved_on_custom_types():
    """The docstring-stripping hook must only remove descriptions derived from the class
    docstring; explicitly configured schema descriptions are preserved."""

    class DocumentedType(dspy.Type):
        """Developer docstring that should not reach the schema."""

        model_config = pydantic.ConfigDict(json_schema_extra={"description": "explicit description"})

        value: str

        def format(self):
            return self.value

    schema = pydantic.TypeAdapter(DocumentedType).json_schema()
    assert schema.get("description") == "explicit description"


def test_docstring_derived_description_is_stripped_from_custom_type_schema():
    class DocstringOnlyType(dspy.Type):
        """Developer docstring that should not reach the schema."""

        value: str

        def format(self):
            return self.value

    schema = pydantic.TypeAdapter(DocstringOnlyType).json_schema()
    assert "description" not in schema
