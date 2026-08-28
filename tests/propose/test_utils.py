from pydantic import BaseModel, Field, create_model

import dspy
from dspy.propose.utils import get_dspy_source_code


class EvidenceMetadata(BaseModel):
    confidence: float = Field(description="Confidence in the extracted evidence")


class EvidenceBase(BaseModel):
    quote: str = Field(description="The exact supporting text")


class Evidence(EvidenceBase):
    metadata: EvidenceMetadata


class ExtractEvidence(dspy.Signature):
    """Extract structured evidence from the input."""

    text: str = dspy.InputField()
    evidence: list[Evidence] | None = dspy.OutputField()
    backup_evidence: Evidence = dspy.OutputField()


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
    assert source.index("class Evidence(EvidenceBase):") < source.index("StringSignature(")


def test_get_dspy_source_code_uses_json_schema_for_dynamic_models():
    source = get_dspy_source_code(DynamicEvidenceModule())

    assert "# JSON Schema for DynamicEvidence" in source
    assert '"value"' in source
    assert "Dynamically generated evidence value" in source
