
import pydantic
import pytest

import dspy
from dspy import Signature


def test_basic_custom_type_resolution():
    """Test basic custom type resolution with both explicit and automatic mapping."""
    class CustomType(pydantic.BaseModel):
        value: str

    # Custom types can be explicitly mapped
    explicit_sig = Signature(
        "input: CustomType -> output: str",
        custom_types={"CustomType": CustomType}
    )
    assert explicit_sig.input_fields["input"].annotation == CustomType

    # Custom types can also be auto-resolved from caller's scope
    auto_sig = Signature("input: CustomType -> output: str")
    assert auto_sig.input_fields["input"].annotation == CustomType


def test_type_alias_for_nested_types():
    """Test using type aliases for nested types."""
    class Container:
        class NestedType(pydantic.BaseModel):
            value: str

    NestedType = Container.NestedType
    alias_sig = Signature("input: str -> output: NestedType")
    assert alias_sig.output_fields["output"].annotation == Container.NestedType

    class Container2:
        class Query(pydantic.BaseModel):
            text: str
        class Score(pydantic.BaseModel):
            score: float

    signature = dspy.Signature("query: Container2.Query -> score: Container2.Score")
    assert signature.output_fields["score"].annotation == Container2.Score


class GlobalCustomType(pydantic.BaseModel):
    """A type defined at module level for testing module-level resolution."""
    value: str
    notes: str = ""


def test_module_level_type_resolution():
    """Test resolution of types defined at module level."""
    # Module-level types can be auto-resolved
    sig = Signature("name: str -> result: GlobalCustomType")
    assert sig.output_fields["result"].annotation == GlobalCustomType


# Create module-level nested class for testing
class OuterContainer:
    class InnerType(pydantic.BaseModel):
        name: str
        value: int


def test_recommended_patterns():
    """Test recommended patterns for working with custom types in signatures."""

    # PATTERN 1: Local type with auto-resolution
    class LocalType(pydantic.BaseModel):
        value: str

    sig1 = Signature("input: str -> output: LocalType")
    assert sig1.output_fields["output"].annotation == LocalType

    # PATTERN 2: Module-level type with auto-resolution
    sig2 = Signature("input: str -> output: GlobalCustomType")
    assert sig2.output_fields["output"].annotation == GlobalCustomType

    # PATTERN 3: Nested type with dot notation
    sig3 = Signature("input: str -> output: OuterContainer.InnerType")
    assert sig3.output_fields["output"].annotation == OuterContainer.InnerType

    # PATTERN 4: Nested type using alias
    InnerTypeAlias = OuterContainer.InnerType
    sig4 = Signature("input: str -> output: InnerTypeAlias")
    assert sig4.output_fields["output"].annotation == InnerTypeAlias

    # PATTERN 5: Nested type with dot notation
    sig5 = Signature("input: str -> output: OuterContainer.InnerType")
    assert sig5.output_fields["output"].annotation == OuterContainer.InnerType

def test_expected_failure():
    # InnerType DNE when not OuterContainer.InnerTypes, so this type shouldnt be resolved
    with pytest.raises(ValueError):
        Signature("input: str -> output: InnerType")

def test_module_type_resolution():
    class TestModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("input: str -> output: OuterContainer.InnerType")

        def predict(self, input: str) -> str:
            return input

    module = TestModule()
    sig = module.predict.signature
    assert sig.output_fields["output"].annotation == OuterContainer.InnerType

def test_basic_custom_type_resolution():
    class CustomType(pydantic.BaseModel):
        value: str

    sig = Signature("input: CustomType -> output: str", custom_types={"CustomType": CustomType})
    assert sig.input_fields["input"].annotation == CustomType

    sig = Signature("input: CustomType -> output: str")
    assert sig.input_fields["input"].annotation == CustomType


def test_explicit_custom_types_do_not_depend_on_the_caller_frame():
    """`custom_types` is documented as the reliable alternative to frame introspection, so it has
    to resolve a type whose *name* is nowhere on the call stack.

    `make_signature` re-parses the signature string a second time to build the default
    instructions, and used to drop `custom_types` on that pass, falling back to the caller-frame
    lookup. The same call then succeeded or raised `Unknown name` depending only on how the caller
    happened to import the type — `from myapp.models import Item` worked, `import myapp.models as
    models` did not.
    """
    # Named only as a string and a dict key: no frame anywhere binds `OffStackItem`.
    item = pydantic.create_model("OffStackItem", value=(str, ...))
    types = {"OffStackItem": item}

    sig = Signature("query: str -> found: OffStackItem", custom_types=types)
    assert sig.output_fields["found"].annotation is item
    assert sig.instructions == "Given the fields `query`, produce the fields `found`."

    nested = Signature("query: str -> found: list[OffStackItem]", custom_types=types)
    assert nested.output_fields["found"].annotation == list[item]

    # Supplying instructions skipped the re-parse entirely; both paths must agree.
    with_instructions = Signature("query: str -> found: OffStackItem", "Find it.", custom_types=types)
    assert with_instructions.output_fields["found"].annotation is item
    assert with_instructions.instructions == "Find it."
