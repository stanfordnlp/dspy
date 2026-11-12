import dspy
from dspy.propose.utils import get_dspy_source_code


class WrittenSignature(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class WrittenProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(WrittenSignature)

    def forward(self, question):
        return self.predict(question=question)


class TwoDynamicSignatures(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.Predict(dspy.Signature("context -> summary"))
        self.answer = dspy.Predict(dspy.Signature("question, summary -> answer"))

    def forward(self, **kwargs):
        raise NotImplementedError


def test_each_dynamic_signature_is_emitted():
    """Every signature built by ``make_signature`` is called ``StringSignature``.

    Keying the deduplication set on that name kept only the first one, so the
    proposer never saw the fields of the others.
    """
    code = get_dspy_source_code(TwoDynamicSignatures())

    assert "context = Field(" in code
    assert "summary = Field(" in code
    assert "question = Field(" in code


def test_repeated_signature_is_emitted_once():
    class Repeated(dspy.Module):
        def __init__(self):
            super().__init__()
            sig = dspy.Signature("question -> answer")
            self.first = dspy.Predict(sig)
            self.second = dspy.Predict(sig)

        def forward(self, **kwargs):
            raise NotImplementedError

    code = get_dspy_source_code(Repeated())

    assert code.count("answer = Field(") == 1


def test_signature_without_parent_namespace_is_handled(monkeypatch):
    """Dynamic signatures do not always carry ``__pydantic_parent_namespace__``.

    The previous code subscripted it unguarded, which raised ``TypeError`` when it
    was ``None`` and ``AttributeError`` when it was absent.
    """
    monkeypatch.setattr(WrittenSignature, "__pydantic_parent_namespace__", None)

    code = get_dspy_source_code(WrittenProgram())

    assert "class WrittenSignature" in code

    monkeypatch.delattr(WrittenSignature, "__pydantic_parent_namespace__")

    code = get_dspy_source_code(WrittenProgram())

    assert "class WrittenSignature" in code


def test_nothing_is_printed(capsys):
    """Reading the source must not write to stdout, it corrupted CLI/notebook output."""
    get_dspy_source_code(WrittenProgram())

    assert capsys.readouterr().out == ""


def test_written_signature_is_read_from_source():
    """A signature declared in a module still contributes its real source, once."""

    class TwoPredictors(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = dspy.Predict(WrittenSignature)
            self.second = dspy.Predict(WrittenSignature)

        def forward(self, **kwargs):
            raise NotImplementedError

    code = get_dspy_source_code(TwoPredictors())

    assert "class WrittenSignature(dspy.Signature):" in code
    assert code.count("class WrittenSignature(dspy.Signature):") == 1
