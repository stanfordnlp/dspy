"""dspy.Flex binds a module's source (``module_src``) and makes it usable like any dspy.Module.

Binding a source — the deterministic baseline at construction, or a GEPA-optimized decomposition —
records it host-side (a syntax check, no interpreter); each forward runs it in a fresh sandbox,
constructing its predictors into that forward's own bridge invocation (nothing is attached to the
module, so parallel forwards can't overwrite each other's predictors). Everything
round-trips through Module.save / load so the generated code can be persisted and rerun. Tests
that run predictors need Deno; pure ``module_src`` checks use a MockInterpreter instead.
(The baseline's RLM-vs-Predict shape is covered in test_tools.py; GEPA's code rewriting in
test_flex_gepa.py.)
"""

from __future__ import annotations

import shutil
import textwrap

import pytest

import dspy
from dspy.flex import Flex
from dspy.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")

# A plain dspy.Predict module we bind and run with a DummyLM, standing in for a GEPA-optimized
# decomposition — end-to-end forward tests bind this instead of the construction baseline.
ECHO_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.echo = dspy.Predict("q -> a")

        def forward(self, q):
            out = self.echo(q=q)
            return dspy.Prediction(a=out.a)
""").strip()


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


@deno_required
def test_forward_attaches_nothing_to_the_module() -> None:
    # A forward's predictors live in that forward's bridge invocation, never on the shared Flex:
    # parallel forwards (e.g. Evaluate(num_threads>1)) run on one module object, and attached
    # predictors would let one forward overwrite another's. The module's surface stays clean,
    # and named_predictors() reports no tunable units — the Flex's update unit is its module_src.
    dspy.configure(lm=DummyLM([{"a": "hi"}]))
    program = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    program(q="hello")
    assert not hasattr(program, "predict")
    assert "predict" not in [n for n, _ in program.named_parameters()]
    assert program.named_predictors() == []


def test_construction_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    # The implementation lives in memory; nothing is written until you call save(). Only module_src is
    # inspected here, so a MockInterpreter keeps it Deno-free.
    monkeypatch.chdir(tmp_path)
    program = Flex(Echo, interpreter_factory=lambda: MockInterpreter())
    assert program.module_src is not None
    assert "dspy.Predict(" in program.module_src
    # Flex itself persists nothing — no module file written until save() is called.
    # (dspy's global `.dspy_cache` may appear; that's unrelated to Flex persistence.)
    assert not list(tmp_path.glob("*.py"))
    assert not list(tmp_path.glob("*.json"))


@deno_required
def test_save_load_roundtrips_generated_code(tmp_path) -> None:
    """Module.save/load carries the generated code (module_src), like instruction state."""
    path = tmp_path / "program.json"
    program = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    program._bind_code(ECHO_MODULE)  # stand in for a GEPA-optimized decomposition
    program.save(path)

    # A fresh Flex starts on the Predict baseline...
    reloaded = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    assert "self.echo" not in reloaded.module_src
    # ...and load() rebinds the saved code (no interpreter or LM needed: the code runs, and
    # constructs its predictors, at each forward).
    reloaded.load(path)
    assert "self.echo" in reloaded.module_src
    assert not hasattr(reloaded, "echo")


@deno_required
def test_nested_flex_state_roundtrips_inside_a_program(tmp_path) -> None:
    """A Flex nested in a larger program persists like a top-level one.

    The parent's state must carry the sub-module's `module_src`, not just its predictors' state:
    those predictors are *created by* that source, so without it a reload has nothing to attach
    them to and the optimized code is lost.
    """
    path = tmp_path / "program.json"

    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sibling = dspy.Predict("x -> y")
            self.flex = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    program = Program()
    program.flex._bind_code(ECHO_MODULE)  # stand in for a GEPA-optimized decomposition
    program.sibling.signature = program.sibling.signature.with_instructions("tuned sibling")

    state = program.dump_state()
    assert state["flex"]["module_src"] == ECHO_MODULE  # the code, not just its predictors
    assert "sibling" in state  # a plain predictor stays a flat top-level entry
    # The Flex's internals belong to its own state; the parent must not dump them separately.
    assert not any(key.startswith("flex.") for key in state)

    program.save(path)

    reloaded = Program()
    assert "self.echo" not in reloaded.flex.module_src  # a fresh program is on the baseline
    reloaded.load(path)

    assert reloaded.flex.module_src == ECHO_MODULE
    assert reloaded.sibling.signature.instructions == "tuned sibling"


def test_flex_is_a_parameter_leaf_in_parent_programs() -> None:
    # Saving, GEPA discovery, and reset_copy all walk named_parameters(); a Flex appears there
    # as one state-bearing leaf (like dspy.Predict), so nothing inside it is enumerated — or
    # optimized — behind the code's back.
    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sibling = dspy.Predict("x -> y")
            self.flex = Flex(Echo, interpreter_factory=lambda: MockInterpreter())

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    program = Program()
    # Even a predictor set directly on the Flex (the bridge itself never attaches any) stays
    # behind the leaf.
    program.flex.echo = dspy.Predict("q -> a")

    params = dict(program.named_parameters())
    assert isinstance(params["flex"], Flex)
    assert not any(name.startswith("flex.") for name in params)
    # The internals are not tunable units from any vantage point; the parent sees one leaf.
    assert program.flex.named_predictors() == []
    assert [n for n, _ in program.named_predictors()] == ["sibling"]


def test_parent_set_lm_and_get_lm_are_symmetric() -> None:
    # set_lm writes to every LM-bearing leaf (Predicts and Flexes); get_lm reads the same set —
    # so set-then-get is an identity, a flex-only program has a unique LM rather than an error,
    # and a divergent flex LM is reported instead of silently ignored.
    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sibling = dspy.Predict("x -> y")
            self.flex = Flex(Echo, interpreter_factory=MockInterpreter)

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    program = Program()
    lm = dspy.LM("openai/gpt-4o-mini")
    program.set_lm(lm)
    assert program.sibling.lm is lm
    assert program.flex.get_lm() is lm
    assert program.get_lm() is lm  # read-your-writes

    program.flex.lm = dspy.LM("openai/gpt-4.1")
    with pytest.raises(ValueError, match="Multiple LMs"):
        program.get_lm()

    class OnlyFlex(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flex = Flex(Echo, interpreter_factory=MockInterpreter)

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    only = OnlyFlex()
    only.set_lm(lm)
    assert only.get_lm() is lm  # previously raised "Multiple LMs" on the empty predictor list


def test_set_lm_is_stored_and_survives_rebind_and_save(tmp_path) -> None:
    # set_lm stores one module-level LM (one home), applied via context at forward time — so it
    # survives rebinds and save/load, unlike writing .lm onto per-forward internals that the
    # next forward rebuilds from scratch.
    flex = Flex(Echo, interpreter_factory=MockInterpreter)
    lm = dspy.LM("openai/gpt-4o-mini")
    flex.set_lm(lm)
    assert flex.get_lm() is lm

    flex._bind_code(ECHO_MODULE)  # a rebind must not lose the module-level LM
    assert flex.get_lm() is lm

    path = tmp_path / "flex.json"
    flex.save(path)
    reloaded = Flex(Echo, interpreter_factory=MockInterpreter)
    reloaded.load(path)
    assert reloaded.get_lm() is not None
    assert reloaded.get_lm().model == lm.model

    # Like Predict: loading a state without an LM clears any previously set one.
    reloaded.load_state({"module_src": ECHO_MODULE, "lm": None})
    assert reloaded.get_lm() is None

    flex.set_lm(lm)
    assert flex.reset_copy().get_lm() is None  # reset clears the LM, like Predict.reset


@deno_required
def test_context_lm_reaches_bridged_calls() -> None:
    # A Flex holds no LM of its own (its state is only module_src); like any module, its
    # predictor calls resolve the ambient LM — including a caller's dspy.context, since the
    # bridged calls run host-side in the calling thread.
    program = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    program._bind_code(ECHO_MODULE)
    with dspy.context(lm=DummyLM([{"a": "from-context-lm"}])):
        assert program(q="hello").a == "from-context-lm"


def test_reset_copy_keeps_optimized_code_and_clears_the_lm() -> None:
    # reset_copy() calls .reset() on every parameter leaf. Predict.reset() clears demos/lm but
    # keeps tuned instructions; a Flex resets the same way — the LM is cleared, the optimized
    # module_src is kept — so optimizer chains (e.g. GEPA then BootstrapFewShot, which compiles
    # a reset copy) don't silently discard the discovered code. There is no other internal state:
    # predictors are rebuilt from the source inside each forward's bridge invocation.
    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flex = Flex(Echo, interpreter_factory=lambda: MockInterpreter())

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    program = Program()
    program.flex._bind_code(ECHO_MODULE)  # stand in for a GEPA-optimized decomposition
    program.flex.set_lm(dspy.LM("openai/gpt-4o-mini"))

    fresh = program.reset_copy()
    assert fresh.flex.module_src == ECHO_MODULE  # the optimized code survives, like instructions
    assert fresh.flex.get_lm() is None  # the LM is cleared, like Predict.reset
    assert program.flex.get_lm() is not None  # the original is untouched


def test_state_layout_unchanged_without_a_flex() -> None:
    # Programs with no Flex keep the flat `{param_path: state}` layout, so existing saved state
    # still loads.
    class Plain(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = dspy.Predict("x -> y")
            self.second = dspy.Predict("y -> z")

        def forward(self, **kwargs):
            return self.second(**self.first(**kwargs))

    assert sorted(Plain().dump_state()) == ["first", "second"]


def test_save_program_roundtrips_via_cloudpickle(tmp_path) -> None:
    # save_program=True cloudpickles the live program object. The bridge (sessions, locks,
    # thread-locals) is excluded from the pickle and rebuilt session-less on load.
    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sibling = dspy.Predict("x -> y")
            self.flex = Flex(Echo, interpreter_factory=MockInterpreter)

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    program = Program()
    program.flex._bind_code(ECHO_MODULE)
    program.save(tmp_path, save_program=True)

    loaded = dspy.load(str(tmp_path), allow_pickle=True)
    assert loaded.flex.module_src == ECHO_MODULE


@deno_required
def test_save_program_end_to_end_forward(tmp_path) -> None:
    """save_program=True with a live Deno session: reload elsewhere and run it."""
    program = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    program._bind_code(ECHO_MODULE)
    program.save(tmp_path, save_program=True)

    dspy.configure(lm=DummyLM([{"a": "echoed-back"}]))
    loaded = dspy.load(str(tmp_path), allow_pickle=True)
    assert loaded.module_src == ECHO_MODULE
    result = loaded(q="hello")
    assert result.a == "echoed-back"


@deno_required
def test_end_to_end_forward_after_save_load(tmp_path) -> None:
    """End-to-end: bind a plain dspy.Predict class, save it, reload into a fresh Flex, run it."""
    path = tmp_path / "program.json"
    program = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    program._bind_code(ECHO_MODULE)
    program.save(path)

    dspy.configure(lm=DummyLM([{"a": "echoed-back"}]))
    reloaded = Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter())
    reloaded.load(path)
    assert "self.echo" in reloaded.module_src
    result = reloaded(q="hello")
    assert isinstance(result, dspy.Prediction)
    assert result.a == "echoed-back"
