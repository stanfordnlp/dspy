"""A dspy.Flex program of several files: its text, and how a forward runs it."""

from __future__ import annotations

import textwrap

import pytest

import dspy
from dspy.predict.flex import program
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.utils.dummies import DummyLM


class Count(dspy.Signature):
    """Count the items in an order."""

    order: str = dspy.InputField()
    total: int = dspy.OutputField()


def text(*files: tuple[str, str]) -> str:
    return program.render({name: textwrap.dedent(source).strip() for name, source in files})


ENTRY = (
    "program.py",
    """
    from .models import Item
    from .rules import total


    class CountModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.Predict("order: str -> names: list[str]")

        def forward(self, **inputs):
            names = self.extract(order=inputs["order"]).names
            return dspy.Prediction(total=total([Item(name) for name in names]))
    """,
)
MODELS = (
    "models.py",
    """
    from dataclasses import dataclass


    @dataclass
    class Item:
        name: str
        quantity: int = 1
    """,
)
RULES = (
    "rules.py",
    """
    def total(items):
        return sum(item.quantity for item in items)
    """,
)
TESTS = (
    "test_rules.py",
    """
    from .models import Item
    from .rules import total


    def test_total_counts_each_item():
        assert total([Item("a"), Item("b", 2)]) == 3
    """,
)


# -- the text --------------------------------------------------------------------------------------------------------


def test_a_text_without_header_lines_is_program_py() -> None:
    source = "class CountModule(dspy.Module):\n    pass"
    assert program.parse(source) == {"program.py": source}
    assert program.render({"program.py": source}) == source


def test_files_follow_their_header_lines_with_program_py_first_and_tests_last() -> None:
    rendered = text(TESTS, RULES, ENTRY, MODELS)
    assert [line for line in rendered.splitlines() if line.startswith("# === ")] == [
        "# === program.py ===",
        "# === models.py ===",
        "# === rules.py ===",
        "# === test_rules.py ===",
    ]
    assert program.render(program.parse(rendered)) == rendered


def test_a_comment_that_is_not_a_file_name_stays_in_its_file() -> None:
    source = "# === helpers ===\ndef f():\n    return 1"
    assert program.parse(source) == {"program.py": source}


@pytest.mark.parametrize(
    "revision",
    [
        "```python\n# === rules.py ===\nX = 1\n```\n\n```python\n# === extra.py ===\nY = 2\n```",
        "Here is the fix:\n```python\n# === rules.py ===\nX = 1\n\n# === extra.py ===\nY = 2\n```",
        "## Changes\n\n# === rules.py ===\nX = 1\n# === extra.py ===\nY = 2",
        "# === rules.py ===\r\nX = 1\r\n# === extra.py ===\r\nY = 2\r\n",
    ],
    ids=["a fence per file", "words and a fence", "a markdown heading", "windows line endings"],
)
def test_a_revision_is_read_without_its_fences_and_words(revision) -> None:
    assert program.read(revision) == {"rules.py": "X = 1", "extra.py": "Y = 2"}


def test_a_fence_inside_the_code_of_a_revision_stays() -> None:
    source = 'EXAMPLE = """\n```json\n{"total": 1}\n```\n"""'
    assert program.read(f"```python\n# === prompts.py ===\n{source}\n```") == {"prompts.py": source}
    assert program.read(f"```python\n{source}\n```") == {"program.py": source}


def test_code_before_the_first_file_of_a_revision_is_program_py_if_it_defines_a_class() -> None:
    revision = "class CountModule(dspy.Module):\n    pass\n\n# === rules.py ===\nX = 1"
    assert program.read(revision) == {"program.py": "class CountModule(dspy.Module):\n    pass", "rules.py": "X = 1"}
    with pytest.raises(CodeInterpreterError, match="twice"):
        program.read(revision + "\n# === program.py ===\nclass CountModule(dspy.Module):\n    pass")


def test_a_revision_replaces_adds_and_removes_files_and_keeps_the_others() -> None:
    before = text(ENTRY, MODELS, RULES)
    revision = "# === rules.py ===\n```python\ndef total(items):\n    return 0\n```\n# === models.py ===\n\n# === extra.py ===\nX = 1"
    after = program.parse(program.revise(before, revision))
    assert sorted(after) == ["extra.py", "program.py", "rules.py"]
    assert after["rules.py"] == "def total(items):\n    return 0"
    assert after["program.py"] == program.parse(before)["program.py"]


def test_words_before_the_files_of_a_revision_are_not_program_py() -> None:
    revision = "Here is the fix: the total counts each item.\n\n# === rules.py ===\ndef total(items):\n    return 0"
    after = program.parse(program.revise(text(ENTRY, RULES), revision))
    assert after["program.py"] == program.parse(text(ENTRY))["program.py"]
    assert after["rules.py"] == "def total(items):\n    return 0"


def test_a_revision_without_header_lines_rewrites_program_py() -> None:
    after = program.parse(program.revise(text(ENTRY, RULES), "class CountModule(dspy.Module):\n    pass"))
    assert after["program.py"].startswith("class CountModule")
    assert "rules.py" in after


def test_a_file_written_twice_is_refused() -> None:
    with pytest.raises(CodeInterpreterError, match=r"rules\.py twice"):
        program.parse("# === rules.py ===\nA = 1\n# === rules.py ===\nA = 2")
    with pytest.raises(CodeInterpreterError, match=r"rules\.py twice"):
        program.read("# === rules.py ===\nA = 1\n# === rules.py ===\nA = 2")


def test_the_entry_is_the_class_named_for_the_flex() -> None:
    source = "class Helper(dspy.Module):\n    def forward(self):\n        pass\n\n\nclass CountModule(dspy.Module):\n    pass"
    assert program.entry({"program.py": source}, "CountModule") == "CountModule"


def test_flex_files_are_the_program_text_parsed() -> None:
    flex = dspy.Flex(Count)
    flex._bind_code(text(ENTRY, MODELS, RULES, TESTS))
    assert sorted(flex.files) == ["models.py", "program.py", "rules.py", "test_rules.py"]
    assert dspy.Flex(Count).files.keys() == {"program.py"}


def test_a_file_that_does_not_parse_does_not_bind() -> None:
    flex = dspy.Flex(Count)
    with pytest.raises(SyntaxError):
        flex._bind_code(text(ENTRY, ("rules.py", "def total(:")))


# -- a forward -------------------------------------------------------------------------------------------------------


@pytest.mark.deno
def test_a_forward_runs_a_program_of_several_files_and_never_imports_its_tests() -> None:
    broken_test = ("test_broken.py", "raise RuntimeError('a test file ran at forward')")
    flex = dspy.Flex(Count)
    flex._bind_code(text(ENTRY, MODELS, RULES, TESTS, broken_test))
    with dspy.context(lm=DummyLM([{"names": ["apple", "pear"]}])):
        assert flex(order="an apple and a pear").total == 2


@pytest.mark.deno
def test_every_file_has_the_tools_and_dspy_in_scope() -> None:
    def shout(text: str) -> str:
        """Uppercase the text."""
        return text.upper()

    class Shout(dspy.Signature):
        """Shout the text."""

        text: str = dspy.InputField()
        loud: str = dspy.OutputField()

    flex = dspy.Flex(Shout, tools=[shout])
    flex._bind_code(
        text(
            ("program.py", """
                from .helpers import louder


                class ShoutModule(dspy.Module):
                    def forward(self, **inputs):
                        return dspy.Prediction(loud=louder(inputs["text"]))
            """),
            ("helpers.py", """
                def louder(text):
                    assert dspy.Predict is not None
                    return shout(text=text) + "!"
            """),
        )
    )
    assert flex(text="hi").loud == "HI!"


@pytest.mark.deno
def test_a_signature_in_another_file_crosses_with_its_types() -> None:
    flex = dspy.Flex(Count)
    flex._bind_code(
        text(
            ("program.py", """
                from .signatures import Extract


                class CountModule(dspy.Module):
                    def __init__(self):
                        super().__init__()
                        self.extract = dspy.Predict(Extract)

                    def forward(self, **inputs):
                        lines = self.extract(order=inputs["order"]).lines
                        return dspy.Prediction(total=sum(line.quantity for line in lines))
            """),
            ("signatures.py", """
                import pydantic


                class Line(pydantic.BaseModel):
                    name: str
                    quantity: int


                class Extract(dspy.Signature):
                    order: str = dspy.InputField()
                    lines: list[Line] = dspy.OutputField()
            """),
        )
    )
    lines = [{"name": "apple", "quantity": 2}, {"name": "pear", "quantity": 3}]
    with dspy.context(lm=DummyLM([{"lines": lines}])):
        assert flex(order="2 apples and 3 pears").total == 5


@pytest.mark.deno
def test_a_program_of_several_files_survives_save_and_load(tmp_path) -> None:
    flex = dspy.Flex(Count)
    flex._bind_code(text(ENTRY, MODELS, RULES, TESTS))
    flex.save(tmp_path / "program.json")

    loaded = dspy.Flex(Count)
    loaded.load(tmp_path / "program.json")
    assert loaded.files == flex.files
    with dspy.context(lm=DummyLM([{"names": ["apple"]}])):
        assert loaded(order="an apple").total == 1


# -- the code proposer -----------------------------------------------------------------------------------------------


def _propose(current: str, *revisions: str) -> tuple[str, DummyLM]:
    """What the proposer makes of ``current`` when the reflection LM writes ``revisions`` in turn."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    lm = DummyLM([{"revised_source": revision} for revision in revisions])
    adapter = DspyAdapter(
        student_module=dspy.Flex(Count),
        metric_fn=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.0,
        feedback_map={},
        reflection_lm=lm,
    )
    reflective = {"self": [{"Inputs": {"order": "x"}, "Generated Outputs": "wrong", "Feedback": "bad"}]}
    return adapter.propose_new_texts({"self": current}, reflective, ["self"])["self"], lm


def test_the_proposer_writes_only_the_files_it_changes() -> None:
    revised = "# === rules.py ===\ndef total(items):\n    return len(items)\n\n# === extra.py ===\n"
    files = program.parse(_propose(text(ENTRY, MODELS, RULES, ("extra.py", "X = 1")), revised)[0])
    assert sorted(files) == ["models.py", "program.py", "rules.py"]
    assert files["rules.py"] == "def total(items):\n    return len(items)"
    assert files["program.py"] == program.parse(text(ENTRY))["program.py"]


def test_a_proposal_without_header_lines_rewrites_program_py() -> None:
    files = program.parse(_propose(text(ENTRY, RULES), "```python\nclass CountModule(dspy.Module):\n    pass\n```")[0])
    assert files["program.py"] == "class CountModule(dspy.Module):\n    pass"
    assert "rules.py" in files


# -- tests -----------------------------------------------------------------------------------------------------------


@pytest.mark.deno
def test_run_tests_reports_each_failure_with_its_file_and_line() -> None:
    wrong = ("test_wrong.py", """
        from .rules import total


        def test_passes():
            assert total([]) == 0


        def test_total_of_nothing_is_one():
            assert total([]) == 1
    """)
    flex = dspy.Flex(Count)
    flex._bind_code(text(ENTRY, MODELS, RULES, TESTS, wrong))
    [failure] = flex.run_tests()
    assert failure.startswith("test_wrong.py::test_total_of_nothing_is_one:")
    assert "test_wrong.py, line 9, in test_total_of_nothing_is_one" in failure
    assert "AssertionError" in failure


@pytest.mark.deno
def test_only_the_test_files_hold_tests_and_a_test_may_add_a_global() -> None:
    helper = ("program.py", program.parse(text(ENTRY))["program.py"] + "\n\n\ndef test_input(x):\n    return x\n")
    adds = ("test_adds.py", """
        def test_adds_a_global():
            globals()["CACHE"] = 1


        def test_runs_after_it():
            assert CACHE == 1
    """)
    flex = dspy.Flex(Count)
    flex._bind_code(text(helper, MODELS, RULES, adds))
    assert flex.run_tests() == []
    assert sorted(flex._bridge.test(flex.module_src)["passed"]) == [
        "test_adds.py::test_adds_a_global", "test_adds.py::test_runs_after_it"
    ]


@pytest.mark.deno
def test_an_async_test_fails_instead_of_passing_unrun() -> None:
    async_test = ("test_async.py", """
        async def test_total_of_nothing_is_one():
            assert False
    """)
    flex = dspy.Flex(Count)
    flex._bind_code(text(ENTRY, MODELS, RULES, async_test))
    [failure] = flex.run_tests()
    assert failure.startswith("test_async.py::test_total_of_nothing_is_one:")
    assert "async test is not run" in failure


@pytest.mark.deno
def test_a_test_may_not_call_the_lm_or_a_tool() -> None:
    def shout(text: str) -> str:
        """Uppercase the text."""
        return text.upper()

    calls = ("test_calls.py", """
        from .program import CountModule


        def test_the_lm():
            CountModule()(order="an apple")


        def test_a_tool():
            shout(text="hi")
    """)
    flex = dspy.Flex(Count, tools=[shout])
    flex._bind_code(text(ENTRY, MODELS, RULES, calls))
    with dspy.context(lm=DummyLM([{"names": ["apple"]}])):
        failures = flex.run_tests()
    assert len(failures) == 2
    assert all("may not call" in failure for failure in failures)


@pytest.mark.deno
def test_a_forward_that_crashes_says_where_in_the_program() -> None:
    from dspy.primitives.code_interpreter import CodeExecutionError

    flex = dspy.Flex(Count)
    flex._bind_code(text(ENTRY, MODELS, ("rules.py", "def total(items):\n    return len(items) // 0")))
    with dspy.context(lm=DummyLM([{"names": ["apple"]}])), pytest.raises(CodeExecutionError) as raised:
        flex(order="an apple")
    message = str(raised.value)
    assert message.startswith("ZeroDivisionError:")
    assert "rules.py, line 2, in total" in message
    assert "program.py, line 12, in forward" in message


@pytest.mark.deno
def test_a_program_written_before_files_still_runs() -> None:
    # As the catalog asked then: helpers nested in forward, whose annotations name each other, and the modules the
    # interpreter had imported used without an import.
    flex = dspy.Flex(Count)
    flex._bind_code(textwrap.dedent("""
        class CountModule(dspy.Module):
            def forward(self, **inputs):
                import pydantic

                class Line(pydantic.BaseModel):
                    name: str
                    quantity: int

                class Extract(dspy.Signature):
                    order: str = dspy.InputField()
                    lines: list[Line] = dspy.OutputField()

                lines = dspy.Predict(Extract)(order=inputs["order"]).lines
                return dspy.Prediction(total=sum(line.quantity for line in lines) + len(json.loads("[]")))
    """))
    with dspy.context(lm=DummyLM([{"lines": [{"name": "apple", "quantity": 2}]}])):
        assert flex(order="2 apples").total == 2


def test_a_test_in_the_hosts_memory_may_not_call_the_lm() -> None:
    from tests.flex.test_flex_interpreter import _ExecInterpreter

    calls = ("test_calls.py", """
        from .program import CountModule


        def test_the_lm():
            CountModule()(order="an apple")
    """)
    flex = dspy.Flex(Count, interpreter_factory=_ExecInterpreter)
    flex._bind_code(text(ENTRY, MODELS, RULES, calls))
    lm = DummyLM([{"names": ["apple"]}])
    with dspy.context(lm=lm):
        [failure] = flex.run_tests()
    assert "may not call the LM" in failure
    assert lm.history == []
