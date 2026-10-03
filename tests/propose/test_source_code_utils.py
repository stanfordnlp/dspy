import dspy
from dspy.propose.utils import get_dspy_source_code


def test_get_dspy_source_code_includes_all_dynamic_signatures():
    qa_signature = dspy.Signature("question -> answer")
    summary_signature = dspy.Signature("text -> summary")

    class TwoDynamicSignatures(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict(qa_signature)
            self.summarize = dspy.Predict(summary_signature)

    code = get_dspy_source_code(TwoDynamicSignatures())
    header = code.split("class TwoDynamicSignatures")[0]

    # Both dynamic signatures must be present, even though both are keyed as
    # "StringSignature" and share the same __name__.
    assert "question -> answer" in header
    assert "text -> summary" in header


def test_get_dspy_source_code_deduplicates_shared_signature():
    qa_signature = dspy.Signature("question -> answer")

    class SharedSignature(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = dspy.Predict(qa_signature)
            self.second = dspy.Predict(qa_signature)

    code = get_dspy_source_code(SharedSignature())
    assert code.count("question -> answer") == 1


def test_get_dspy_source_code_does_not_print_signatures(capsys):
    class Predictor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question -> answer")

    get_dspy_source_code(Predictor())
    assert capsys.readouterr().out == ""


def test_get_dspy_source_code_handles_none_parent_namespace():
    # On Python 3.14, pydantic leaves __pydantic_parent_namespace__ as None for
    # dynamically created signatures (see #9937). The dedup key must not
    # subscript it.
    qa_signature = dspy.Signature("question -> answer")
    qa_signature.__pydantic_parent_namespace__ = None

    class NoneNamespaceSignature(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(qa_signature)

    code = get_dspy_source_code(NoneNamespaceSignature())
    assert "question -> answer" in code
