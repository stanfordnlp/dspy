from dataclasses import dataclass, make_dataclass
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, create_model

import dspy
from dspy.propose.utils import get_dspy_source_code


class EvidenceMetadata(BaseModel):
    confidence: float = Field(description="Confidence in the extracted evidence")


class EvidenceBase(BaseModel):
    quote: str = Field(description="The exact supporting text")


class Evidence(EvidenceBase):
    metadata: EvidenceMetadata


class EvidenceKind(Enum):
    QUOTE = "quote"


class OptionalOnlyKind(Enum):
    SUMMARY = "summary"


class CallableOnlyKind(Enum):
    FACT = "fact"


class NestedAliasOnlyKind(Enum):
    CLAIM = "claim"


@dataclass
class EvidenceLocation:
    page: int


EvidenceKinds = list[EvidenceKind]
MaybeEvidenceKind = OptionalOnlyKind | None
EvidenceKindFormatter = Callable[[CallableOnlyKind], str]
NestedEvidenceKinds = list[NestedAliasOnlyKind]
NestedEvidenceKindMap = dict[str, NestedEvidenceKinds]
UnusedEvidenceKinds = tuple[EvidenceKind, ...]

RuntimeEvidenceKind = Enum("RuntimeEvidenceKind", {"QUOTE": "quote"})
RuntimeEvidenceLocation = make_dataclass("RuntimeEvidenceLocation", [("page", int)])


class TypedEvidence(BaseModel):
    kind: EvidenceKind
    location: EvidenceLocation
    allowed_kinds: EvidenceKinds
    maybe_kind: MaybeEvidenceKind
    formatter: EvidenceKindFormatter
    nested_kinds: NestedEvidenceKindMap

    def unrelated_helper(self):
        return UnusedEvidenceKinds


class RuntimeTypedEvidence(BaseModel):
    kind: RuntimeEvidenceKind
    location: RuntimeEvidenceLocation


class ExtractEvidence(dspy.Signature):
    """Extract structured evidence from the input."""

    text: str = dspy.InputField()
    evidence: list[Evidence] | None = dspy.OutputField()
    backup_evidence: Evidence = dspy.OutputField()
    typed_evidence: TypedEvidence = dspy.OutputField()
    runtime_typed_evidence: RuntimeTypedEvidence = dspy.OutputField()


class EvidenceModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(ExtractEvidence)

    def forward(self, text: str):
        return self.predictor(text=text)


DynamicEvidence = create_model(
    "DynamicEvidence",
    value=(str, Field(description="Dynamically generated evidence value")),
)


class ExtractDynamicEvidence(dspy.Signature):
    text: str = dspy.InputField()
    evidence: DynamicEvidence = dspy.OutputField()


class DynamicEvidenceModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(ExtractDynamicEvidence)


def test_get_dspy_source_code_includes_referenced_pydantic_models():
    source = get_dspy_source_code(EvidenceModule())

    assert "class EvidenceMetadata(BaseModel):" in source
    assert "confidence: float" in source
    assert "class EvidenceBase(BaseModel):" in source
    assert "quote: str" in source
    assert "class Evidence(EvidenceBase):" in source
    assert "metadata: EvidenceMetadata" in source
    assert source.count("class Evidence(EvidenceBase):") == 1
    assert source.index("class EvidenceBase(BaseModel):") < source.index(
        "class Evidence(EvidenceBase):"
    )
    assert source.index("class EvidenceMetadata(BaseModel):") < source.index(
        "class Evidence(EvidenceBase):"
    )
    assert "class EvidenceKind(Enum):" in source
    assert "class Enum(metaclass=EnumType):" not in source
    assert "class Flag(Enum" not in source
    assert "class EvidenceLocation:" in source
    assert "EvidenceKinds = list[EvidenceKind]" in source
    assert "MaybeEvidenceKind = OptionalOnlyKind | None" in source
    assert "class OptionalOnlyKind(Enum):" in source
    assert "EvidenceKindFormatter = Callable[[CallableOnlyKind], str]" in source
    assert "class CallableOnlyKind(Enum):" in source
    assert "NestedEvidenceKinds = list[NestedAliasOnlyKind]" in source
    assert "NestedEvidenceKindMap = dict[str, NestedEvidenceKinds]" in source
    assert "class NestedAliasOnlyKind(Enum):" in source
    assert source.index("class NestedAliasOnlyKind(Enum):") < source.index(
        "NestedEvidenceKinds = list[NestedAliasOnlyKind]"
    )
    assert source.index("NestedEvidenceKinds = list[NestedAliasOnlyKind]") < source.index(
        "NestedEvidenceKindMap = dict[str, NestedEvidenceKinds]"
    )
    assert "UnusedEvidenceKinds = tuple[EvidenceKind, ...]" not in source
    assert "RuntimeEvidenceKind = Enum(" in source
    assert "class RuntimeEvidenceLocation:" in source
    assert source.count("RuntimeEvidenceKind = Enum(") == 1
    assert source.count("class RuntimeEvidenceLocation:") == 1
    assert source.index("class EvidenceKind(Enum):") < source.index(
        "class TypedEvidence(BaseModel):"
    )
    assert source.index("class EvidenceLocation:") < source.index(
        "class TypedEvidence(BaseModel):"
    )
    assert source.index("EvidenceKinds = list[EvidenceKind]") < source.index(
        "class TypedEvidence(BaseModel):"
    )
    assert source.index("class Evidence(EvidenceBase):") < source.index("StringSignature(")


def test_get_dspy_source_code_uses_json_schema_for_dynamic_models():
    source = get_dspy_source_code(DynamicEvidenceModule())

    assert "# JSON Schema for DynamicEvidence" in source
    assert '"value"' in source
    assert "Dynamically generated evidence value" in source
