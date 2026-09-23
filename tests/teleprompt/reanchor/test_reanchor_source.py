"""A signature written back as Python source."""

import ast
from typing import Annotated, Literal

import pytest

import dspy
from dspy.experimental import Choice, Noul, Score
from dspy.teleprompt.reanchor.source import render_signature

Attention = Score["Ignore it", "Read it", "Reply to it"]
Action = Choice[("reply", "I answer it."), ("read", "I open it.")]


class Triage(dspy.Signature):
    """Triage an email."""

    sender: str = dspy.InputField(desc="Name and address")
    body: str = dspy.InputField()
    needs_response: bool = dspy.OutputField(desc="Does this need a reply?")
    attention: Annotated[float, Attention] = dspy.OutputField(desc="How much attention")
    action: Action = dspy.OutputField(desc="What I will do")
    folder: Literal["inbox", "later"] = dspy.OutputField()


def test_render_writes_the_docstring_inputs_and_rubrics():
    src = render_signature(Triage)
    assert src.startswith('class Triage(dspy.Signature):\n    """Triage an email."""')
    assert "sender: str = dspy.InputField(desc='Name and address')" in src
    assert "body: str = dspy.InputField()" in src
    assert "needs_response: bool = dspy.OutputField(desc='Does this need a reply?')" in src
    assert "attention: Annotated[float, Score['Ignore it', 'Read it', 'Reply to it']]" in src
    assert "action: Choice[('reply', 'I answer it.'), ('read', 'I open it.')]" in src
    assert "folder: Literal['inbox', 'later'] = dspy.OutputField()" in src
    ast.parse(src)


def test_a_long_rubric_is_written_one_option_per_line():
    long = Score[
        "A notification or newsletter I will never open.",
        "Something I will read once.",
        "A message from a person that I will answer.",
    ]
    sig = dspy.Signature({"body": (str, dspy.InputField()), "attention": (long, dspy.OutputField(desc="Q"))}, "T")
    src = render_signature(sig, name="Sig")
    assert "attention: Score[\n        'A notification" in src and "\n    ] = dspy.OutputField(desc='Q')" in src
    ast.parse(src)


def test_a_rich_noul_without_descriptions_is_written_bare():
    class Flag(dspy.Signature):
        text: str = dspy.InputField()
        flag: Noul = dspy.OutputField(desc="Is it relevant?")

    assert "flag: Noul = dspy.OutputField(desc='Is it relevant?')" in render_signature(Flag)


def test_a_typed_noul_writes_true_before_false():
    blocked_type = Noul[(False, "Workaround exists."), (True, "Service down.")]

    class Outage(dspy.Signature):
        ticket: str = dspy.InputField()
        blocked: Annotated[bool, blocked_type] = dspy.OutputField(desc="Is the service blocked?")

    src = render_signature(Outage)
    assert "blocked: Annotated[bool, Noul[(True, 'Service down.'), (False, 'Workaround exists.')]]" in src


@pytest.mark.parametrize(
    "text",
    [
        'Triage an email that ends with a quote"',
        'Triage an email with a """ literal inside',
        "Triage using a literal backslash-t sequence: C:\\temp",
        "Triage an email.\n\nRead the whole thread before deciding.",
        '""""',
        '"',
        "Triage an email that ends with a backslash\\",
    ],
)
def test_a_tricky_docstring_is_written_as_valid_python(text):
    tree = ast.parse(render_signature(Triage.with_instructions(text), name="Triage"))
    assert ast.get_docstring(tree.body[0], clean=False) == text
