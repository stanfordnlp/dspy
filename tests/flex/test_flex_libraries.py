"""dspy.Flex code written with the user's own libraries: what the optimizer is told, where the code runs, the
versions saved with it, and the tool calls a metric can score."""

import importlib
import logging
import pickle
import shutil
import sys
import textwrap

import cloudpickle
import pytest

import dspy
from dspy.primitives import sandbox_bundle
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")

# A library no model has seen, as a module of the user's own project: on the path, but not installed.
TALLYHO = '''
"""tallyho: a small ledger SDK that keeps amounts in minor units."""

import json

__version__ = "0.3.1"


class Ledger:
    """An in-memory ledger of signed amounts in cents."""

    def __init__(self, currency: str = "EUR") -> None:
        self.currency = currency
        self._entries = []

    def post(self, memo: str, cents: int) -> int:
        """Record an entry and return its id."""
        self._entries.append((memo, cents))
        return len(self._entries)

    def balance(self) -> int:
        """The sum of all entries, in cents."""
        return sum(cents for _, cents in self._entries)

    def _dump(self) -> str:
        return json.dumps(self._entries)


def parse_amount(text: str) -> int:
    """Parse a European amount like '12,50 €' into cents.

    Thousands separators are dots.
    """
    return round(float(text.replace("€", "").replace(".", "").replace(",", ".").strip()) * 100)
'''

TOTAL = textwrap.dedent(
    """
    class TotalModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            import flextallyho

            ledger = flextallyho.Ledger()
            for line in inputs["lines"].splitlines():
                ledger.post(line, flextallyho.parse_amount(line.split(":")[1]))
            return dspy.Prediction(total_cents=ledger.balance())
    """
)

LINES = "coffee: 3,20 €\nrefund: -1.000,00 €"

# A start for code that only uses the libraries, for the optimizer to write.
STUB = textwrap.dedent(
    """
    class TotalModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            raise NotImplementedError("Not written yet: implement forward with the libraries.")
    """
)


@pytest.fixture
def tallyho(tmp_path, monkeypatch):
    (tmp_path / "flextallyho.py").write_text(TALLYHO)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("flextallyho")
    sys.modules.pop("flextallyho", None)


@pytest.fixture
def warnings_seen(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("dspy"), "propagate", True)  # so that caplog sees dspy's warnings
    caplog.set_level(logging.WARNING)
    return caplog


def _total(tallyho, **kwargs):
    return dspy.Flex("lines: str -> total_cents: int", libraries=[tallyho], **kwargs)


def test_a_library_is_documented_from_its_own_api(tallyho):
    rendered = dspy.Library(tallyho, docs="Post refunds as negative amounts.").render()

    assert rendered.startswith("### flextallyho 0.3.1")
    assert "class flextallyho.Ledger(currency: str = 'EUR')\n" in rendered
    assert "    def post(memo: str, cents: int) -> int\n        # Record an entry and return its id." in rendered
    assert (
        "def flextallyho.parse_amount(text: str) -> int\n    # Parse a European amount like '12,50 €' into cents.\n"
        "    #\n    # Thousands separators are dots."  # the whole docstring
    ) in rendered
    assert rendered.endswith("Post refunds as negative amounts.")
    assert "_dump" not in rendered and "flextallyho.json" not in rendered  # private, and what it imported


def test_members_choose_what_is_documented_and_the_reference_is_capped(tallyho):
    chosen = dspy.Library(tallyho, members=["parse_amount"]).reference()
    assert "parse_amount" in chosen and "Ledger" not in chosen

    capped = dspy.Library(tallyho, max_chars=80).reference()
    assert len(capped) < 200 and capped.endswith("Pass members=[...] to choose what is documented.")


def test_the_optimizer_is_told_about_the_libraries_next_to_dspy_modules(tallyho):
    flex = _total(tallyho)
    context = flex._flex_ctx.render_context_blurb(sandboxed=True)

    assert dspy.Library(tallyho).reference() in context
    assert "DSPy's modules for what needs a language model" in context and "dspy.ReAct" in context
    assert "dspy.Predict" in flex.module_src  # the usual baseline
    assert _total(tallyho, module_src=STUB).module_src == STUB  # or code of your own, like a library-only start


def test_the_code_proposer_reads_the_library_reference(tallyho):
    reflection = DummyLM([{"revised_source": TOTAL}])
    student = _total(tallyho, module_src=STUB)
    adapter = DspyAdapter(student_module=student, metric_fn=lambda *a, **k: 0.0, feedback_map={}, reflection_lm=reflection)
    reflective = {"self": [{"Inputs": {"lines": LINES}, "Generated Outputs": "NotImplementedError", "Feedback": "x"}]}

    out = adapter.propose_new_texts({"self": student.module_src}, reflective, ["self"])

    assert out["self"].strip() == TOTAL.strip()
    prompt = "\n".join(message["content"] for message in reflection.history[-1]["messages"])
    assert "def flextallyho.parse_amount(text: str) -> int" in prompt  # the reference
    assert "Common patterns" in prompt  # and DSPy's own catalog


def test_library_code_runs_in_a_local_interpreter(tallyho):
    # The worker runs Python in isolated mode, so it finds a project module only from where the host imports it.
    flex = _total(tallyho, module_src=TOTAL, interpreter_factory=dspy.LocalInterpreter)
    assert flex(lines=LINES).total_cents == -99680


@deno_required
def test_library_code_runs_in_the_pyodide_sandbox(tallyho):
    flex = _total(tallyho, module_src=TOTAL, interpreter_factory=dspy.PythonInterpreter)
    assert flex(lines=LINES).total_cents == -99680


def test_a_compiled_library_is_refused_for_the_sandbox(monkeypatch):
    monkeypatch.setattr(sandbox_bundle, "_pyodide_lock", lambda deno_dir: {})
    monkeypatch.setattr(sandbox_bundle, "_distributions", lambda extra=(): [])
    monkeypatch.setattr(sandbox_bundle, "_module_files", lambda top, dist=None: None)  # as for a compiled module
    with pytest.raises(sandbox_bundle.SandboxBundleError, match="cannot run in the Pyodide sandbox"):
        sandbox_bundle._plan.__wrapped__(None, ("fastledger",))


def test_a_saved_program_warns_when_its_libraries_or_tools_change(tallyho, monkeypatch, warnings_seen):
    def lookup(order_id: int) -> dict:
        """Look up an order."""
        return {}

    flex = _total(tallyho, module_src=TOTAL, tools=[lookup])
    state = flex.dump_state()
    pickled = cloudpickle.dumps(flex)
    assert state["environment"]["libraries"]["flextallyho"]["version"] == "0.3.1"

    _total(tallyho, tools=[lookup]).load_state(state)
    pickle.loads(pickled)
    assert not warnings_seen.records  # the same environment

    monkeypatch.setattr(tallyho.Ledger, "void", lambda self, entry_id: None, raising=False)  # the API, not the version
    _total(tallyho, tools=[lookup]).load_state(state)
    assert "flextallyho's API changed" in warnings_seen.text
    warnings_seen.clear()
    pickle.loads(pickled)
    assert "flextallyho's API changed" in warnings_seen.text

    warnings_seen.clear()
    monkeypatch.setattr(tallyho, "__version__", "0.4.0")

    def other_lookup(order_id: int) -> dict:
        """Look up an order, with its items."""
        return {}

    _total(tallyho, tools=[dspy.Tool(other_lookup, name="lookup")]).load_state(state)
    assert "flextallyho is 0.4.0, not 0.3.1" in warnings_seen.text
    assert "tool lookup changed" in warnings_seen.text


def test_the_tool_calls_of_a_forward_are_on_its_prediction():
    def lookup(order_id: int) -> dict:
        """Look up an order."""
        return {"id": order_id, "paid": 1250}

    def refund(order_id: int, cents: int) -> str:
        """Refund an order."""
        raise RuntimeError("refunds are closed")

    source = textwrap.dedent(
        """
        class Support(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                order = lookup(order_id=inputs["order_id"])
                try:
                    refund(order_id=order["id"], cents=order["paid"])
                except Exception:
                    pass
                return dspy.Prediction(reply="done")
        """
    )
    flex = dspy.Flex("order_id: int -> reply", tools=[lookup, refund], module_src=source,
                     interpreter_factory=dspy.LocalInterpreter)

    prediction = flex(order_id=7)

    # Side effects a metric can score, and explain in its feedback. They are not outputs.
    assert prediction.tool_calls == [
        {"name": "lookup", "args": {"order_id": 7}, "result": {"id": 7, "paid": 1250}},
        {"name": "refund", "args": {"order_id": 7, "cents": 1250}, "error": "RuntimeError: refunds are closed"},
    ]
    assert "tool_calls" not in prediction.keys()


@deno_required
def test_a_pooled_interpreter_has_the_libraries_it_was_given(tallyho):
    with dspy.InterpreterPool(size=1, libraries=["flextallyho"]) as pool:
        flex = _total(tallyho, module_src=TOTAL, interpreter_factory=pool)
        assert flex(lines=LINES).total_cents == -99680
        restored = pickle.loads(cloudpickle.dumps(pool))
        try:
            assert restored._libraries == ("flextallyho",)
        finally:
            restored.close()


@deno_required
def test_an_interpreter_takes_libraries_until_it_starts(tallyho):
    interpreter = dspy.PythonInterpreter()
    interpreter.provide_dspy()
    interpreter.provide_dspy(libraries=["flextallyho"])  # not started yet: its bundle is replaced
    assert sum(arg.startswith("--dspy-bundle=") for arg in interpreter.deno_command) == 1
    with interpreter:
        assert interpreter.execute("import flextallyho\nflextallyho.parse_amount('1,00 €')") == 100
        interpreter.provide_dspy(libraries=["flextallyho"])  # nothing new: nothing to do
        with pytest.raises(sandbox_bundle.SandboxBundleError, match="started without json5"):
            interpreter.provide_dspy(libraries=["json5"])


def test_gepa_writes_the_module_from_an_unimplemented_start(tallyho):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        got = getattr(pred, "total_cents", None)
        return dspy.Prediction(score=float(got == gold.total_cents), feedback=f"expected {gold.total_cents}, got {got}")

    examples = [dspy.Example(lines=f"a: {n},00 €\nb: 1,50 €", total_cents=n * 100 + 150).with_inputs("lines")
                for n in range(1, 4)]
    student = _total(tallyho, module_src=STUB, interpreter_factory=dspy.LocalInterpreter)
    reflection = DummyLM([{"revised_source": TOTAL}] * 10)

    optimized = dspy.GEPA(metric=metric, reflection_lm=reflection, max_metric_calls=12).compile(
        student, trainset=examples, valset=examples
    )

    assert "NotImplementedError" in str(reflection.history[0]["messages"])  # the stub's failure, as feedback
    assert optimized(lines="x: 2,00 €\ny: 0,99 €").total_cents == 299
