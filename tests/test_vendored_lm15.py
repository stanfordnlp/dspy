"""The vendored lm15 copy must import as ``dspy._vendor.lm15`` from any install."""

from pathlib import Path


def test_vendored_lm15_imports_under_dspy_namespace():
    from dspy._vendor import lm15
    from dspy._vendor.lm15 import Message, Request
    from dspy._vendor.lm15.providers import openai

    assert "_vendor" in Path(lm15.__file__).parts
    assert openai.__name__ == "dspy._vendor.lm15.providers.openai"
    assert Request(model="x", messages=[Message.user("hi")]).model == "x"


def test_vendored_lm15_carries_provenance_and_license():
    import dspy._vendor.lm15 as lm15

    pkg = Path(lm15.__file__).parent
    marker = (pkg / "VENDORED").read_text()
    assert "commit=" in marker and "digest=" in marker
    assert "MIT" in (pkg / "LICENSE").read_text()
