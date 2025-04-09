import pytest
from unittest.mock import patch
from pydantic_core import ValidationError
from pydantic import TypeAdapter
import dspy
from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature
from dspy.utils.dummies import DummyLM

class MockSignature(Signature):
    question: str = dspy.InputField()
    answer: int = dspy.OutputField()

def test_adapter_retry_logic():
    adapter = Adapter()
    lm = DummyLM([{"answer": "42"}])
    demos = []
    inputs = {"question": "6 x 7"}

    with dspy.context(adapter_retry_count=1, lm=lm):
        # Mock parse to raise ValidationError on the first call and succeed on the second
        with patch.object(adapter, "format", return_value="formatted_input") as mock_format, \
            patch.object(adapter, "parse", side_effect=lambda signature, completion: TypeAdapter(signature).validate_python(completion)) as mock_parse:
            with pytest.raises(ValidationError):
                adapter(lm, {}, MockSignature, demos, inputs)

    assert mock_format.call_count == 2
    _, kwargs = mock_format.call_args
    assert kwargs["inputs"]["previous_response"] == '["[[ ## answer ## ]]\\n42"]'
    assert kwargs["inputs"]["question"] == "6 x 7"
    assert mock_parse.call_count == 2
    assert lm.call_count == 2
