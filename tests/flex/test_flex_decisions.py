"""dspy.Flex with decision outputs (Noul, Score, Choice) on a System One client.

Decision values cross the sandbox with their host semantics, sub-signature strings can declare
decision types, code can configure a predictor's decision questions through ``fields``, and the
Flex's ``predictor_fields`` (what ReAnchor fits) apply on top. The baseline predictor gets the
Flex's declared signature, so its questions and input descriptions reach the backend intact.
Most tests use ``dspy.LocalInterpreter``, which runs the same shim without needing Deno.
"""

import shutil
import textwrap
from typing import Literal

import pytest

import dspy
from dspy.experimental import Choice, Score
from dspy.predict.flex.ctx import DECISION_NOTE
from dspy.primitives.code_interpreter import CodeInterpreterError
from tests.teleprompt.reanchor.fakes import FakeClient, choice, noul, score

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")

Level = Score["low", "medium", "high"]
Kind = Choice[("billing", "Payment issue"), ("technical", "Product bug")]


class Triage(dspy.Signature):
    """Triage a support ticket."""

    ticket: str = dspy.InputField(desc="The customer's message")
    urgent: bool = dspy.OutputField(desc="Is the customer blocked?")
    severity: Level = dspy.OutputField(desc="How bad is it?")
    kind: Kind = dspy.OutputField(desc="What kind of issue?")


def answer(state, name, q):
    """Fixed evidence per question type, so each test can check how it was decoded."""
    if q["type"] == "noul":
        return noul(0.7)
    if q["type"] == "score":
        return score({0: 0.1, 1: 0.3, 2: 0.6})
    return choice({"billing": 0.4, "technical": 0.6})


@pytest.fixture
def client():
    client = FakeClient(answer)
    dspy.configure(lm=client)
    return client


def flex(signature=Triage, source=None, **kwargs):
    program = dspy.Flex(signature, interpreter_factory=dspy.LocalInterpreter, **kwargs)
    if source is not None:
        program._bind_code(textwrap.dedent(source).strip())
    return program


def test_the_baseline_predictor_gets_the_declared_signature(client):
    result = flex()(ticket="The app crashes")
    state, questions = client.calls[0]
    # Field descriptions, which the rendered signature string cannot carry, reach the backend.
    assert state["instructions"] == "Triage a support ticket."
    assert "The customer's message" in state["input_fields"]
    assert questions["urgent"]["instructions"] == "Is the customer blocked?"
    assert questions["severity"]["criteria"] == ["low", "medium", "high"]
    assert result.urgent is True
    assert isinstance(result.severity, Level) and result.severity.level == 2
    assert isinstance(result.kind, Kind) and result.kind.value == "technical"


def test_the_baseline_renders_decision_types_in_its_signature_string():
    source = flex().module_src
    assert "severity: Score['low', 'medium', 'high']" in source
    assert "kind: Choice[('billing', 'Payment issue'), ('technical', 'Product bug')]" in source


def test_decision_values_keep_their_host_semantics_in_the_sandbox(client):
    program = flex(
        source="""
        class Checks(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.Predict(dspy.Signature(
                    "ticket -> blocked: Noul, level: Score['low', 'medium', 'high'], "
                    "area: Choice[('billing', 'Payment'), ('technical', 'Bug')]",
                    "Assess the ticket.",
                ))
                self.judge.fields["blocked"] = {"instructions": "Is the customer blocked?"}
                self.judge.fields["level"] = {"instructions": "How bad is it?"}
                self.judge.fields["area"] = {"instructions": "What kind of issue?"}

            def forward(self, ticket):
                out = self.judge(ticket=ticket)
                assert bool(out.blocked) is True and out.blocked.value is True
                assert out.blocked.probability == 0.7
                assert abs(float(out.level) - 1.5) < 1e-9 and out.level.level == 2
                assert out.level.probabilities[2] == 0.6  # keyed by level index, as on the host
                assert out.area.value == "technical" and out.area.probabilities["billing"] == 0.4
                return dspy.Prediction(urgent=out.blocked, severity=out.level, kind=out.area)
        """
    )
    result = program(ticket="The app crashes")
    # A decision returned for a native output becomes its value; for a decision output, the declared type.
    assert result.urgent is True
    assert isinstance(result.severity, Level) and result.severity.probabilities == {0: 0.1, 1: 0.3, 2: 0.6}
    assert isinstance(result.kind, Kind) and result.kind.confidence == 0.7


class Urgent(dspy.Signature):
    ticket: str = dspy.InputField()
    urgent: bool = dspy.OutputField(desc="Is the customer blocked?")
    severity: Level | None = dspy.OutputField(desc="How bad is it?")
    kind: Literal["billing", "technical"] = dspy.OutputField(desc="What kind of issue?")


def test_decisions_passed_on_arrive_as_decision_objects(client):
    seen = []

    def log_severity(severity) -> str:
        seen.append(severity)
        return "logged"

    program = flex(
        Urgent,
        tools=[log_severity],
        source="""
        class Relay(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.Predict("ticket -> level: Score['low', 'medium', 'high']")
                self.judge.fields["level"] = {"instructions": "How bad is it?"}

            def forward(self, ticket):
                out = self.judge(ticket=ticket)
                log_severity(severity=out.level)
                return dspy.Prediction(urgent=out.level.level == 2, severity=[out.level][0], kind="billing")
        """,
    )
    program(ticket="The app crashes")
    assert isinstance(seen[0], Score) and seen[0].level == 2


def test_a_decision_output_without_a_question_fails_with_the_reason(client):
    program = flex(
        source="""
        class NoQuestion(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.Predict("ticket -> blocked: Noul")

            def forward(self, ticket):
                return dspy.Prediction(urgent=self.judge(ticket=ticket).blocked, severity=None, kind="billing")
        """
    )
    with pytest.raises(CodeInterpreterError, match="requires an OutputField"):
        program(ticket="The app crashes")


CODE_FIELDS = """
class Coded(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.Predict("ticket -> blocked: bool")
        self.judge.fields["blocked"] = {"instructions": "Is the customer blocked?", "threshold": 0.9}

    def forward(self, ticket):
        blocked = self.judge(ticket=ticket).blocked
        return dspy.Prediction(urgent=blocked, severity=None, kind="billing")
"""


def test_code_fields_configure_the_question_and_predictor_fields_override_them(client):
    program = flex(Urgent, source=CODE_FIELDS)
    assert program(ticket="x").urgent is False  # 0.7 is below the code's threshold of 0.9
    assert client.calls[-1][1]["blocked"]["instructions"] == "Is the customer blocked?"

    program.predictor_fields = {"judge": {"blocked": {"threshold": 0.6}}}
    assert program(ticket="x").urgent is True  # The stored threshold replaces the code's.
    assert client.calls[-1][1]["blocked"]["instructions"] == "Is the customer blocked?"  # The code's question stays.


def test_invalid_predictor_fields_name_the_predictor(client):
    program = flex(Urgent, source=CODE_FIELDS)
    program.predictor_fields = {"judge": {"blocked": {"threshold": 2}}}
    with pytest.raises(CodeInterpreterError, match=r"predictor 'judge' \(from the Flex's predictor_fields\)"):
        program(ticket="x")


def test_chain_of_thought_fields_configure_its_inner_predict(client):
    program = flex(
        Urgent,
        source="""
        class Reasoned(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.ChainOfThought("ticket -> blocked: bool")
                self.judge.fields["blocked"] = {"instructions": "Is the customer blocked?"}

            def forward(self, ticket):
                return dspy.Prediction(urgent=self.judge(ticket=ticket).blocked, severity=None, kind="billing")
        """,
    )
    program.predictor_fields = {"judge.predict": {"blocked": {"threshold": 0.8}}}
    with pytest.raises(CodeInterpreterError, match="Unsupported System One output 'reasoning'"):
        program(ticket="x")  # System One answers decisions only; the reasoning field has no slot.


def test_predictor_fields_round_trip_and_clear_when_the_code_changes(tmp_path):
    program = flex(Urgent, source=CODE_FIELDS)
    assert "predictor_fields" not in program.dump_state()
    program.predictor_fields = {"judge": {"blocked": {"threshold": 0.6}}, "unused": {}}
    assert program.dump_state()["predictor_fields"] == {"judge": {"blocked": {"threshold": 0.6}}}

    path = tmp_path / "flex.json"
    program.save(path)
    loaded = flex(Urgent)
    loaded.load(path)
    assert loaded.module_src == program.module_src
    assert loaded.predictor_fields == {"judge": {"blocked": {"threshold": 0.6}}}

    loaded._bind_code(loaded.module_src)  # The same code keeps what was fitted to it.
    assert loaded.predictor_fields == {"judge": {"blocked": {"threshold": 0.6}}}
    loaded._bind_code(loaded.module_src.replace("Coded", "Recoded"))
    assert loaded.predictor_fields == {}


def test_loading_malformed_predictor_fields_fails():
    program = flex(Urgent)
    with pytest.raises(ValueError, match="predictor_fields must map"):
        program.load_state({"module_src": program.module_src, "predictor_fields": {"judge": 0.5}})


def test_the_code_proposer_learns_about_decision_outputs():
    assert DECISION_NOTE in flex()._flex_ctx.render_context_blurb(sandboxed=True)

    class Plain(dspy.Signature):
        q: str = dspy.InputField()
        a: bool = dspy.OutputField()

    assert DECISION_NOTE not in flex(Plain)._flex_ctx.render_context_blurb(sandboxed=True)


@deno_required
def test_decisions_cross_the_deno_sandbox(client):
    program = dspy.Flex(Urgent, interpreter_factory=dspy.PythonInterpreter)
    program._bind_code(
        textwrap.dedent("""
        class Gate(dspy.Module):
            def __init__(self):
                super().__init__()
                self.judge = dspy.Predict("ticket -> blocked: Noul")
                self.judge.fields["blocked"] = {"instructions": "Is the customer blocked?"}

            def forward(self, ticket):
                out = self.judge(ticket=ticket)
                kind = "technical" if out.blocked and out.blocked.probability > 0.6 else "billing"
                return dspy.Prediction(urgent=out.blocked, severity=None, kind=kind)
        """).strip()
    )
    result = program(ticket="The app crashes")
    assert result.urgent is True and result.kind == "technical"
