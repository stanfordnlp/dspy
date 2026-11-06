import dspy
from dspy.propose.utils import get_dspy_source_code


def test_get_dspy_source_code_with_dynamic_signature():
    """Dynamically created signatures can leave ``__pydantic_parent_namespace__`` set
    to ``None`` (for example on Python 3.14), and extracting their source code must not
    crash with ``TypeError: 'NoneType' object is not subscriptable``."""

    class DynamicProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(dspy.Signature("question -> answer"))

    program = DynamicProgram()
    # Reproduce the state where pydantic leaves the parent namespace unset.
    program.predict.signature.__pydantic_parent_namespace__ = None

    source = get_dspy_source_code(program)
    assert isinstance(source, str)


def test_get_dspy_source_code_with_class_signature():
    class QASignature(dspy.Signature):
        """Answer the question."""

        question = dspy.InputField()
        answer = dspy.OutputField()

    class QAProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(QASignature)

    source = get_dspy_source_code(QAProgram())
    assert isinstance(source, str)
    assert "QASignature" in source
