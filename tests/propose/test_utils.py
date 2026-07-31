import dspy
from dspy.propose.utils import get_dspy_source_code


class TwoDynamicSignatures(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict(dspy.Signature("question -> answer"))
        self.summarize = dspy.Predict(dspy.Signature("text -> summary"))


def test_get_dspy_source_code_keeps_distinct_dynamic_signatures():
    # Two separately constructed dynamic signatures share the same
    # __pydantic_parent_namespace__["signature_name"] ("Signature"), since that
    # value comes from SignatureMeta.__new__'s parameter name, not the
    # signature's own content. Deduping on that string alone would drop the
    # second signature; deduping on object identity must not.
    code = get_dspy_source_code(TwoDynamicSignatures())
    header = code.split("class TwoDynamicSignatures")[0]

    assert "-> answer" in header
    assert "-> summary" in header


class RepeatedSignatureReference(dspy.Module):
    def __init__(self):
        super().__init__()
        shared = dspy.Signature("question -> answer")
        self.first = dspy.Predict(shared)
        self.second = dspy.Predict(shared)


def test_get_dspy_source_code_dedupes_repeated_signature_reference():
    # Two Predictors referencing the *same* signature object should still
    # only contribute one copy of that signature's source to the header.
    code = get_dspy_source_code(RepeatedSignatureReference())
    header = code.split("class RepeatedSignatureReference")[0]

    assert header.count("-> answer") == 1
