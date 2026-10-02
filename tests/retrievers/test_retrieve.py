import dspy
from dspy.dsp.utils.utils import dotdict
from dspy.retrievers.retrieve import Retrieve


def test_retrieve_accepts_string_passages():
    dspy.configure(rm=lambda query, k=3, **kwargs: ["alpha", "beta"][:k])
    result = Retrieve(k=2)("q")
    assert result.passages == ["alpha", "beta"]


def test_retrieve_accepts_a_single_string_passage():
    dspy.configure(rm=lambda query, k=3, **kwargs: "only")
    result = Retrieve(k=1)("q")
    assert result.passages == ["only"]


def test_retrieve_still_reads_long_text_objects():
    dspy.configure(rm=lambda query, k=3, **kwargs: [dotdict(long_text="doc")])
    result = Retrieve(k=1)("q")
    assert result.passages == ["doc"]
