"""dspy.Flex(interpreter_factory=...) runs generated code, with the real dspy, inside an interpreter.

* The interpreter_factory is validated.
* Behavior on LocalInterpreter (a subprocess, no Deno) and the Deno/Pyodide sandbox: predictors run in the
  sandbox while every LM call runs on the host, which builds the request from its own LM, admits only
  generation options, meters the call, and reports LM errors by type; provided tools are dspy.Tools there.
* Integration (Deno required, skipped otherwise): optimizer-authored glue runs inside a
  ``dspy.PythonInterpreter``, and adversarial file access is contained to the sandbox.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import decimal
import enum
import inspect
import json
import shutil
import sys
import textwrap
import typing

import pydantic
import pytest

import dspy
from dspy.predict.flex import Flex, bridge
from dspy.primitives.code_interpreter import (
    CodeExecutionError,
    CodeInterpreterError,
    resolve_interpreter_factory,
)
from dspy.primitives.sandbox_guest import encode_value
from dspy.signatures.signature import make_signature
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMError, LMRateLimitError
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


class Doubler(dspy.Signature):
    """Return double the input value."""

    value: int = dspy.InputField()
    result: int = dspy.OutputField()


def shout(text: str) -> str:
    """Uppercase helper used to exercise user tools through the bridge."""
    return text.upper()


class ShoutSig(dspy.Signature):
    """Uppercase the input text."""

    text: str = dspy.InputField()
    out: str = dspy.OutputField()


# =============================================================================
# Normalization / construction (no Deno)
# =============================================================================


def test_default_interpreter_factory_is_python_interpreter() -> None:
    # PythonInterpreter is the public default. The resolver lets a configured factory override it.
    default = inspect.signature(Flex.__init__).parameters["interpreter_factory"].default
    assert default is dspy.PythonInterpreter


@deno_required
def test_default_is_sandbox() -> None:
    # With no interpreter_factory argument, Flex builds a PythonInterpreter sandbox (needs Deno).
    flex = Flex(Doubler)
    assert flex._bridge is not None
    assert flex._interpreter_factory is dspy.PythonInterpreter
    assert isinstance(resolve_interpreter_factory(flex._interpreter_factory)(), dspy.PythonInterpreter)


def test_bare_instance_is_rejected() -> None:
    # There is no in-process mode and no shared-instance mode: like dspy.RLM, the constructor takes a
    # factory, not a live interpreter. Passing an instance is a TypeError.
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter_factory=MockInterpreter())  # type: ignore[arg-type]


def test_none_interpreter_factory_is_rejected() -> None:
    # None once meant "run in-process"; that path is gone, so it remains invalid.
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter_factory=None)  # type: ignore[arg-type]


def test_factory_is_used_as_is() -> None:
    flex = Flex(Doubler, interpreter_factory=lambda: MockInterpreter())
    assert flex._bridge is not None
    assert callable(flex._interpreter_factory)


def test_bad_interpreter_type_raises() -> None:
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter_factory=123)  # type: ignore[arg-type]


def test_tools_plus_interpreter_is_supported() -> None:
    flex = Flex(Doubler, tools=[shout], interpreter_factory=lambda: MockInterpreter())
    assert flex._bridge is not None
    # the user tool is registered (name -> callable) so sandbox glue can call it by name
    registered = flex._bridge._tool_callables()["shout"]
    assert registered.__name__ == "shout" and registered(text="hi") == shout(text="hi")


# =============================================================================
# The real dspy in the sandbox, wired to the host (LocalInterpreter, and Deno when installed)
# =============================================================================
# Sandboxed code imports the real dspy; every LM call it makes crosses to the host, which builds the
# request from its own LM, admits only generation options, meters the call, and runs it.

BACKENDS = [
    pytest.param(dspy.LocalInterpreter, id="local"),
    pytest.param(dspy.PythonInterpreter, marks=deno_required, id="deno"),
]


def _module(body: str, init: str = "") -> str:
    lines = ["class DoublerModule(dspy.Module):", "    def __init__(self):", "        super().__init__()"]
    lines += ["        " + line for line in textwrap.dedent(init).splitlines() if line]
    lines += ["", "    def forward(self, value):"]
    lines += ["        " + line for line in textwrap.dedent(body).splitlines()]
    return "\n".join(lines)


@pytest.mark.parametrize("factory", BACKENDS)
def test_predictors_run_in_the_sandbox_on_the_host_lm(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        "return dspy.Prediction(result=self.solve(value=value).result)",
        init='self.solve = dspy.ChainOfThought("value: int -> result: int")',
    ))
    lm = DummyLM([{"plan": "two doubled is four", "result": "4"}])
    with dspy.context(lm=lm):
        assert flex(value=2).result == 4
    assert len(lm.history) == 1  # the host LM made the call
    # Nothing is attached to the Flex: its internals belong to the code.
    assert not hasattr(flex, "solve")
    assert flex.named_predictors() == []


@pytest.mark.parametrize("factory", BACKENDS)
def test_sandboxed_code_cannot_set_lm_routing_or_credentials(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        'out = dspy.Predict("value: int -> result: int")(value=value, config={"api_base": "http://evil.example"})\n'
        "return dspy.Prediction(result=out.result)"
    ))
    lm = DummyLM([{"result": "4"}])
    with dspy.context(lm=lm), pytest.raises(CodeExecutionError, match=r"may not set LM option\(s\) \['api_base'\]"):
        flex(value=2)
    assert lm.history == []


@pytest.mark.parametrize("factory", [*BACKENDS, pytest.param(lambda: _ExecInterpreter(), id="in-process")])
def test_sandboxed_code_cannot_change_the_request_settings_the_host_owns(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        "import dataclasses\n"
        "lm = dspy.settings.lm\n"
        'request = lm._build_adapter_request("hi", {"temperature": 0.5})\n'
        'config = dataclasses.replace(request.config, extensions={"custom_llm_provider": "anthropic"})\n'
        "lm(dataclasses.replace(request, config=config))\n"
        "return dspy.Prediction(result=0)"
    ))
    lm = DummyLM([{"result": "4"}])
    with dspy.context(lm=lm), pytest.raises(Exception, match=r"may not change request setting\(s\) \['extensions'\]"):
        flex(value=2)
    assert lm.history == []


@pytest.mark.parametrize("factory", BACKENDS)
def test_sandboxed_code_cannot_switch_the_model(factory) -> None:
    # The host serves every request with its own LM; a request naming another model is refused.
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        "import dataclasses\n"
        "lm = dspy.settings.lm\n"
        'request = dataclasses.replace(lm._build_adapter_request("hi", {}), model="openai/gpt-5-pro")\n'
        "lm(request)\n"
        "return dspy.Prediction(result=0)"
    ))
    lm = DummyLM([{"result": "4"}])
    with dspy.context(lm=lm), pytest.raises(CodeExecutionError, match=r"Request\.model must match LM\.model"):
        flex(value=2)
    assert lm.history == []


@pytest.mark.parametrize("factory", BACKENDS)
def test_sandboxed_code_cannot_have_the_host_read_its_files(factory, tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        "from pathlib import Path\n"
        "from dspy._vendor.lm15.types import DocumentPart\n"
        "lm = dspy.settings.lm\n"
        f"part = DocumentPart(media_type='text/plain', path=Path({str(secret)!r}))\n"
        "lm(dspy.lm15.Request(model=lm.model, messages=(dspy.lm15.Message.user((part,)),)))\n"
        "return dspy.Prediction(result=0)"
    ))
    lm = DummyLM([{"result": "1"}])
    with dspy.context(lm=lm), pytest.raises(CodeExecutionError, match="may not reference host files"):
        flex(value=1)
    assert lm.history == []


@pytest.mark.parametrize("factory", BACKENDS)
def test_lm_call_budget_is_enforced_per_forward(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory, max_predictor_calls=2)
    flex._bind_code(_module(
        'p = dspy.Predict("value: int -> result: int")\n'
        "for _ in range(int(value)):\n"
        "    out = p(value=value)\n"
        "return dspy.Prediction(result=out.result)"
    ))
    with dspy.context(lm=DummyLM([{"result": "1"}] * 10)):
        assert flex(value=2).result == 1
        with pytest.raises(CodeInterpreterError, match=r"budget \(2\).*Raise max_predictor_calls if this is expected"):
            flex(value=3)
        assert flex(value=2).result == 1  # each forward starts with a fresh budget


def test_lm_call_budget_can_be_disabled() -> None:
    flex = Flex(Doubler, interpreter_factory=dspy.LocalInterpreter, max_predictor_calls=None)
    assert flex._bridge._max_predictor_calls is None
    flex._bind_code(_module(
        'p = dspy.Predict("value: int -> result: int")\n'
        "for _ in range(5):\n"
        "    out = p(value=value)\n"
        "return dspy.Prediction(result=out.result)"
    ))
    with dspy.context(lm=DummyLM([{"result": "1"}] * 5)):
        assert flex(value=1).result == 1


@pytest.mark.parametrize("factory", BACKENDS)
def test_provided_tools_are_dspy_tools_with_their_host_metadata(factory) -> None:
    # Modules built in the sandbox see the host tool's name, description and arguments, and generated code
    # can still call it like a function, positionally or by keyword.
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=factory)
    flex._bind_code(textwrap.dedent(
        """
        class ShoutModule(dspy.Module):
            def forward(self, text):
                assert isinstance(shout, dspy.Tool)
                assert (shout.name, shout.desc) == ("shout", "Uppercase helper used to exercise user tools through the bridge.")
                assert shout.args == {"text": {"type": "string"}}
                return dspy.Prediction(out=shout(text) + shout(text="!"))
        """
    ).strip())
    assert flex(text="hi").out == "HI!"


@pytest.mark.parametrize("factory", BACKENDS)
def test_react_in_the_sandbox_calls_a_host_tool_and_a_sandbox_helper(factory) -> None:
    calls = []

    def lookup(query: str) -> str:
        """Look things up."""
        calls.append(query)
        return f"found {query}"

    flex = Flex(ShoutSig, tools=[lookup], interpreter_factory=factory)
    flex._bind_code(textwrap.dedent(
        """
        class AgentModule(dspy.Module):
            def forward(self, text):
                def reverse(s: str) -> str:
                    \"\"\"Reverse a string.\"\"\"
                    return s[::-1]

                agent = dspy.ReAct("question: str -> answer: str", tools=[lookup, reverse])
                return dspy.Prediction(out=agent(question=text).answer)
        """
    ).strip())
    lm = DummyLM([
        {"next_thought": "look", "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}])},
        {"next_thought": "flip", "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "reverse", "args": {"s": "abc"}}])},
        {"next_thought": "done", "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "cba"}}])},
    ])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        assert flex(text="cats").out == "cba"
    assert calls == ["cats"]  # the host tool ran on the host; the helper never left the sandbox
    assert "Look things up." in lm.history[0]["request"].messages[0].text  # the host tool's description


def _recording(factory):
    """``factory``, recording every interpreter it makes."""
    made = []

    def make():
        made.append(factory())
        return made[-1]

    make.execution_instructions = factory.execution_instructions
    return make, made


@pytest.mark.parametrize("factory", BACKENDS)
def test_code_executing_modules_get_a_fresh_interpreter_from_the_flex_factory(factory) -> None:
    # The nested RLM's REPL is a separate interpreter: what its code does to builtins, sys.modules or globals
    # does not reach the module that built it. It still calls back into that module's functions and budget.
    make, made = _recording(factory)
    flex = Flex(Doubler, interpreter_factory=make, max_predictor_calls=3)
    flex._bind_code(_module(
        "import builtins, sys\n"
        "def halve(n: int) -> int:\n"
        '    """Halve n."""\n'
        "    return n // 2\n"
        'rlm = dspy.RLM("value: int -> result: int", max_iters=2, tools=[halve])\n'
        "out = rlm(value=value).result\n"
        "leaked = hasattr(builtins, 'leaked') or 'leaked' in sys.modules or 'leaked' in globals()\n"
        "return dspy.Prediction(result=-1 if leaked else out)"
    ))
    code = (
        "```python\nimport builtins, sys\nbuiltins.leaked = sys.modules['leaked'] = leaked = True\n"
        "SUBMIT(result=halve(n=value) + (2 if llm_query('two?') else 0))\n```"
    )
    lm = DummyLM([{"plan": "halve it", "code": code}, {"response": "2"}])
    with dspy.context(lm=lm):
        assert flex(value=40).result == 22
    assert len(made) == 2  # the Flex's own interpreter, and one for the nested RLM
    assert len(lm.history) == 2  # the nested RLM's action and its llm_query, both on the Flex's budget


def test_nested_interpreters_are_shut_down_with_the_forward() -> None:
    shut = []

    class Tracked(dspy.LocalInterpreter):
        def shutdown(self):
            shut.append(self)
            super().shutdown()

    make, made = _recording(Tracked)
    flex = Flex(Doubler, interpreter_factory=make)
    flex._bind_code(_module(
        'repl = dspy.RLM("value: int -> result: int", max_iters=1)\n'
        "return dspy.Prediction(result=repl(value=value).result)"
    ))
    with dspy.context(lm=DummyLM([{"plan": "p", "code": "SUBMIT(result=value)"}])):
        assert flex(value=3).result == 3
    assert len(made) == 2 and set(map(id, shut)) == set(map(id, made))


@pytest.mark.parametrize("factory", BACKENDS)
def test_host_adapter_is_used_in_the_sandbox(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    lm = DummyLM([{"result": "4"}], adapter=dspy.JSONAdapter())
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        assert flex(value=2).result == 4
    assert "JSON" in lm.history[-1]["request"].system


class _ShoutingAdapter(dspy.ChatAdapter):
    def format_task_description(self, signature):
        return "SHOUTING: " + super().format_task_description(signature)


@pytest.mark.parametrize("factory", BACKENDS)
def test_a_custom_adapter_is_used_in_the_sandbox(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    lm = DummyLM([{"result": "4"}])
    with dspy.context(lm=lm, adapter=_ShoutingAdapter()):
        assert flex(value=2).result == 4
    assert "SHOUTING:" in lm.history[-1]["request"].system


@pytest.mark.parametrize("factory", BACKENDS)
def test_inner_predictor_calls_join_the_host_trace(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        "return dspy.Prediction(result=self.solve(value=value).result)",
        init='self.solve = dspy.ChainOfThought(dspy.Signature("value: int -> result: int", "Double it."))',
    ))
    with dspy.context(lm=DummyLM([{"plan": "p", "result": "4"}]), trace=[]):
        flex(value=2)
        [(predictor, inputs, prediction)] = dspy.settings.trace
    assert predictor.signature.instructions == "Double it."
    assert predictor.signature.input_fields["value"].annotation is int
    assert predictor.signature.output_fields["reasoning"].annotation is dspy.Reasoning
    assert inputs == {"value": 2}
    assert prediction.result == 4


class _Address(pydantic.BaseModel):
    postal_code: str = pydantic.Field(serialization_alias="postalCode")
    city: str = pydantic.Field(alias="cityName")


class _Where(dspy.Signature):
    address: _Address = dspy.InputField()
    code: str = dspy.OutputField()
    same: _Address = dspy.OutputField()


@pytest.mark.parametrize("factory", [dspy.LocalInterpreter, lambda: _ExecInterpreter()], ids=["local", "in-process"])
def test_models_cross_as_models_whatever_their_aliases(factory) -> None:
    flex = Flex(_Where, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, address):\n"
        "        return dspy.Prediction(code=address.postal_code, same=address)\n"
    )
    address = _Address(postal_code="10001", cityName="New York")
    out = flex(address=address)
    assert (out.code, out.same) == ("10001", address)


class _Account(pydantic.BaseModel):
    name: str
    token: str = pydantic.Field(default="", exclude=True)


class _Owner(dspy.Signature):
    account: _Account = dspy.InputField()
    seen: str = dspy.OutputField()


@pytest.mark.parametrize("factory", BACKENDS)
def test_excluded_input_fields_do_not_reach_generated_code(factory) -> None:
    flex = Flex(_Owner, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, account):\n"
        "        return dspy.Prediction(seen=account.name + ':' + account.token)\n"
    )
    assert flex(account=_Account(name="a", token="secret")).seen == "a:"


def _tagged(allowed: set[str]) -> type[dspy.Signature]:
    class Tag(pydantic.BaseModel):
        name: str

        @pydantic.field_validator("name")
        @classmethod
        def _known(cls, value):
            if value not in allowed:
                raise ValueError(f"not one of {sorted(allowed)}")
            return value

    # Built from the types themselves: this file's annotations are strings, which can't name a local class.
    return make_signature({"tag": (Tag, dspy.InputField()), "name": (str, dspy.OutputField())})


@pytest.mark.parametrize("factory", BACKENDS)
def test_inputs_of_a_type_defined_in_a_function_reach_generated_code_as_that_type(factory) -> None:
    signature = _tagged({"red"})
    flex = Flex(signature, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, tag):\n"
        "        return dspy.Prediction(name=f'{type(tag).__name__} {tag.name}')\n"
    )
    assert flex(tag=signature.input_fields["tag"].annotation(name="red")).name == "Tag red"


class _Placed(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)
    coordinates: tuple[int, int]
    data: bytes = b""


class _Where2(dspy.Signature):
    place: _Placed = dspy.InputField()
    report: str = dspy.OutputField()
    same: _Placed = dspy.OutputField()


@pytest.mark.parametrize("factory", BACKENDS)
def test_strict_models_cross_as_the_json_pydantic_wrote(factory) -> None:
    flex = Flex(_Where2, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, place):\n"
        "        return dspy.Prediction(report=f'{place.coordinates!r} {place.data!r}', same=place)\n"
    )
    place = _Placed(coordinates=(1, 2), data=b"hi")
    out = flex(place=place)
    assert (out.report, out.same) == ("(1, 2) b'hi'", place)


def test_the_optimizer_is_told_the_users_types_are_data_in_the_sandbox() -> None:
    blurb = Flex(_Owner)._flex_ctx.render_context_blurb(sandboxed=True)
    assert "In the sandbox, _Account are plain data classes" in blurb and "don't call their methods" in blurb
    assert "plain data classes" not in Flex("q -> a")._flex_ctx.render_context_blurb(sandboxed=True)


class _Color(str, enum.Enum):
    RED = "red"
    BLUE = "blue"


class _Point(pydantic.BaseModel, frozen=True):
    x: int


class _Shape(pydantic.BaseModel):
    color: _Color
    points: set[_Point]
    groups: set[frozenset[int]]
    span: tuple[int, ...]
    data: bytes
    amount: decimal.Decimal
    tags: set[str] = pydantic.Field(default_factory=set)


class _Size(pydantic.RootModel[int]):
    pass


@dataclasses.dataclass
class _Frame:
    width: int
    tags: set[str] = dataclasses.field(default_factory=set)


class _Corner(typing.NamedTuple):
    x: int
    y: int = 0


class _Canvas(pydantic.BaseModel):
    size: _Size
    frame: _Frame
    corner: _Corner


class _Paint(dspy.Signature):
    canvas: _Canvas = dspy.InputField()
    report: str = dspy.OutputField()


@pytest.mark.parametrize("factory", BACKENDS)
def test_root_models_dataclasses_and_named_tuples_reach_generated_code(factory) -> None:
    flex = Flex(_Paint, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, canvas):\n"
        "        import dataclasses\n"
        "        frame = dataclasses.replace(canvas.frame, width=canvas.frame.width + 1)\n"
        "        return dspy.Prediction(report=f'{canvas.size.root} {frame.width} {canvas.corner.x} {canvas.corner.y}')\n"
    )
    assert flex(canvas=_Canvas(size=_Size(3), frame=_Frame(4), corner=_Corner(5))).report == "3 5 5 0"


@pydantic.dataclasses.dataclass
class _Place:
    postal_code: str = pydantic.Field(alias="postal code")


class _Locate(dspy.Signature):
    text: str = dspy.InputField()
    place: _Place = dspy.OutputField()


@pytest.mark.parametrize("factory", BACKENDS)
def test_dataclass_outputs_parse_under_their_alias(factory) -> None:
    key = "postal code"  # as on the host, which reads this field by its alias only
    flex = Flex(_Locate, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, text):\n"
        "        return dspy.Prediction(place=dspy.Predict('text -> place: _Place')(text=text).place)\n"
    )
    with dspy.context(lm=DummyLM([{"place": json.dumps({key: "10001"})}])):
        assert flex(text="t").place.postal_code == "10001"


class _Nested(pydantic.BaseModel):
    x: int = pydantic.Field(validation_alias=pydantic.AliasPath("data", "x"))


class _Unwrap(dspy.Signature):
    value: int = dspy.InputField()
    nested: _Nested = dspy.OutputField()


@pytest.mark.parametrize("factory", [*BACKENDS, pytest.param(lambda: _ExecInterpreter(), id="in-process")])
def test_models_with_alias_paths_return_to_the_host(factory) -> None:
    flex = Flex(_Unwrap, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, value):\n"
        "        return dspy.Prediction(nested=_Nested.model_validate({'data': {'x': value}}))\n"
    )
    assert flex(value=7).nested.x == 7


class _Tone(str, enum.Enum):
    WARM = "warm"
    COOL = "cool"


class _Swatch(pydantic.BaseModel):
    tone: _Tone


class _Mark(pydantic.BaseModel):
    name: str
    id: str = pydantic.Field(default_factory=lambda: "host-made")


class _Match(dspy.Signature):
    swatch: _Swatch = dspy.InputField()
    wanted: _Tone = dspy.InputField()
    same: bool = dspy.OutputField()
    mark: _Mark = dspy.OutputField()


@pytest.mark.parametrize("factory", BACKENDS)
def test_shared_types_compare_equal_and_host_factories_make_defaults(factory) -> None:
    flex = Flex(_Match, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, swatch, wanted):\n"
        "        return dspy.Prediction(same=swatch.tone == wanted and swatch.tone is wanted, mark=_Mark(name='m'))\n"
    )
    out = flex(swatch=_Swatch(tone=_Tone.COOL), wanted=_Tone.COOL)
    assert (out.same, out.mark.id) == (True, "host-made")


class _Sides(pydantic.BaseModel):  # both fields' alias choices include the same name
    left: int = pydantic.Field(validation_alias=pydantic.AliasChoices(pydantic.AliasPath("left", "value"), "value"))
    right: int = pydantic.Field(validation_alias=pydantic.AliasChoices(pydantic.AliasPath("right", "value"), "value"))


class _Compare(dspy.Signature):
    sides: _Sides = dspy.InputField()
    diff: int = dspy.OutputField()
    same: _Sides = dspy.OutputField()


@pytest.mark.parametrize("factory", [*BACKENDS, pytest.param(lambda: _ExecInterpreter(), id="in-process")])
def test_fields_whose_alias_choices_share_a_name_keep_their_own_values(factory) -> None:
    flex = Flex(_Compare, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, sides):\n"
        "        return dspy.Prediction(diff=sides.left - sides.right, same=sides)\n"
    )
    sides = _Sides.model_validate({"left": {"value": 1}, "right": {"value": 2}})
    out = flex(sides=sides)
    assert (out.diff, out.same) == (-1, sides)


class _Draw(dspy.Signature):
    shape: _Shape = dspy.InputField()
    report: str = dspy.OutputField()
    same: _Shape = dspy.OutputField()


@pytest.mark.parametrize("factory", BACKENDS)
def test_model_inputs_keep_their_python_types_in_generated_code(factory) -> None:
    flex = Flex(_Draw, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, shape):\n"
        "        shape.tags.add('seen')\n"
        "        kinds = [type(getattr(shape, name)).__name__ for name in ('points', 'groups', 'span', 'data', 'amount')]\n"
        "        report = ' '.join([shape.color.name, type(next(iter(shape.points))).__name__, *kinds])\n"
        "        return dspy.Prediction(report=report, same=shape)\n"
    )
    shape = _Shape(color=_Color.BLUE, points={_Point(x=1)}, groups={frozenset({1})}, span=(1, 2), data=b"ab",
                   amount=decimal.Decimal("1.5"))
    out = flex(shape=shape)
    assert out.report == "BLUE _Point set set tuple bytes Decimal"
    assert out.same == shape.model_copy(update={"tags": {"seen"}})


class _Order(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(ser_json_bytes="hex", val_json_bytes="hex")
    items: pydantic.Json[list[int]]
    secret: str = pydantic.Field(default="", exclude=True)
    marks: set[str] = pydantic.Field(default_factory=set, exclude=True)
    note: bytes = b"ab"


class _Ship(pydantic.BaseModel):
    order: _Order
    count: int = pydantic.Field(validation_alias=pydantic.AliasChoices("n", "count_"))


class _Pack(dspy.Signature):
    ship: _Ship = dspy.InputField()
    report: str = dspy.OutputField()


@pytest.mark.parametrize("factory", [*BACKENDS, pytest.param(lambda: _ExecInterpreter(), id="in-process")])
def test_json_fields_exclusions_and_model_defaults_reach_generated_code(factory) -> None:
    flex = Flex(_Pack, interpreter_factory=factory)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, ship):\n"
        "        fresh = type(ship.order).model_fields['note'].default\n"
        "        ship.order.marks.add('seen')\n"
        "        return dspy.Prediction(report=f'{sum(ship.order.items)} {ship.count} {fresh!r}')\n"
    )
    ship = _Ship(order=_Order(items="[1, 2]", secret="s"), count_=3)
    assert flex(ship=ship).report == "3 3 b'ab'"



    main = DummyLM([{"result": "four, I think"}], adapter=dspy.ChatAdapter())
    extraction = DummyLM([{"result": "4"}], adapter=dspy.ChatAdapter())
    flex = Flex(Doubler, interpreter_factory=dspy.LocalInterpreter)
    with dspy.context(lm=main, adapter=dspy.TwoStepAdapter(extraction)):
        assert flex(value=2).result == 4
    assert len(main.history) == len(extraction.history) == 1


@pytest.mark.parametrize("factory", BACKENDS)
def test_each_best_of_n_attempt_reaches_the_host_with_its_own_rollout(factory) -> None:
    seen = []

    class Recording(DummyLM):
        def __call__(self, prompt=None, cache=None, rollout_id=None, **options):
            seen.append(rollout_id)
            return super().__call__(prompt, cache=cache, rollout_id=rollout_id, **options)

    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(_module(
        "return dspy.Prediction(result=self.best(value=value).result)",
        init='self.best = dspy.BestOfN(dspy.Predict("value: int -> result: int"), N=3, '
             "reward_fn=lambda args, pred: 0.0, threshold=1.0)",
    ))
    with dspy.context(lm=Recording([{"result": "4"}] * 3)):
        flex(value=2)
    assert seen == [0, 1, 2]  # so a cached first attempt does not answer the others


def test_best_of_n_winning_attempt_joins_the_host_trace() -> None:
    flex = Flex(Doubler, interpreter_factory=dspy.LocalInterpreter)
    flex._bind_code(_module(
        "return dspy.Prediction(result=self.best(value=value).result)",
        init='self.best = dspy.BestOfN(dspy.Predict("value: int -> result: int"), N=2, '
             "reward_fn=lambda args, pred: 1.0, threshold=1.0)",
    ))
    with dspy.context(lm=DummyLM([{"result": "4"}] * 2), trace=[]):
        assert flex(value=2).result == 4
        [(_, inputs, prediction)] = dspy.settings.trace
    assert (inputs, prediction.result) == ({"value": 2}, 4)


def test_parallel_forwards_run_in_their_own_sandboxes() -> None:
    from concurrent.futures import ThreadPoolExecutor

    flex = Flex(Doubler, interpreter_factory=dspy.LocalInterpreter)
    flex._bind_code(_module("return dspy.Prediction(result=value * 2)"))
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(lambda v: flex(value=v).result, range(8))) == [v * 2 for v in range(8)]


def test_interpreter_is_not_serialized() -> None:
    flex = Flex(Doubler, interpreter_factory=dspy.LocalInterpreter)
    state = flex.dump_state()
    assert state.get("module_src")  # the code persists
    assert "interpreter" not in json.dumps(state)  # the runtime sandbox does not


# =============================================================================
# bridge.py helpers (no Deno)
# =============================================================================


def test_parse_module_class_name_prefers_forward() -> None:
    src = textwrap.dedent(
        """
        class Helper:
            pass

        class MyModule(dspy.Module):
            def forward(self, **inputs):
                return dspy.Prediction()
        """
    )
    assert bridge.parse_module_class_name(src) == "MyModule"


def test_outputs_cross_in_json_mode() -> None:
    # Everything dspy.Predict can output must cross the boundary: datetimes, enums, and pydantic
    # models holding them serialize in JSON mode, and dspy types cross as their fields.
    class Color(enum.Enum):
        RED = "red"

    class Event(pydantic.BaseModel):
        when: datetime.datetime
        color: Color

    fields = encode_value({
        "event": Event(when=datetime.datetime(2026, 1, 1), color=Color.RED),
        "when": datetime.datetime(2026, 1, 2),
        "color": Color.RED,
        "image": dspy.Image(url="https://example.com/x.png"),
    })
    assert fields["event"] == {"when": "2026-01-01T00:00:00", "color": "red"}
    assert fields["when"] == "2026-01-02T00:00:00"
    assert fields["color"] == "red"
    assert fields["image"] == {"url": "https://example.com/x.png"}
    json.dumps(fields)


class _ExecInterpreter:
    """Runs code with exec() in the host process, in a private namespace; leaves sys.modules alone."""

    def __init__(self):
        self.tools = {}
        self._namespace = {}

    def start(self) -> None:
        pass

    def execute(self, code: str, variables: dict | None = None):
        self._namespace.update(self.tools, **(variables or {}))
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, "<flex>", "exec"), self._namespace)
        return eval(compile(ast.Expression(last.value), "<flex>", "eval"), self._namespace) if last else None

    def shutdown(self) -> None:
        self._namespace.clear()


def test_in_process_interpreter_uses_the_host_dspy_on_the_metered_lm() -> None:
    def louder(text: str) -> str:
        """Uppercase the text."""
        return text.upper()

    flex = Flex(ShoutSig, tools=[louder], interpreter_factory=_ExecInterpreter, max_predictor_calls=1)
    flex._bind_code(
        textwrap.dedent(
            """
            class Shout(dspy.Module):
                def __init__(self):
                    self.draft = dspy.Predict("text -> out")

                def forward(self, text):
                    out = self.draft(text=text).out
                    if text == "twice":
                        self.draft(text=text)
                    return dspy.Prediction(out=louder(text=out))
            """
        )
    )
    with dspy.context(lm=DummyLM([{"out": "hello"}] * 3)):
        assert flex(text="hi").out == "HELLO"
        with pytest.raises(CodeInterpreterError, match="budget"):
            flex(text="twice")
    assert sys.modules["dspy"] is dspy
    assert dspy.settings.lm is None  # the host's settings are untouched


# =============================================================================
# LM infrastructure errors keep their type across the boundary
# =============================================================================
# The boundary flattens a sandbox exception into a message string, so a provider failure would surface as a
# generic CodeExecutionError. The host stashes the typed error and stamps a tag into the message that crosses
# the boundary; forward() re-raises the typed error only when the failure that killed the run carries that
# tag. Infrastructure failures are not the generated code's fault, and callers discriminate by type (e.g.
# catching dspy.LMError to retry) — but a failure the generated code recovered from must not repaint a later,
# unrelated crash as an infrastructure error.


def _rate_limited_lm(_answers=None):
    """A host LM whose provider is down: every call raises a typed LM infrastructure error."""
    from tests.test_utils.engines import failing_lm

    return failing_lm(LMRateLimitError("429 from the provider", model="dummy"))


LM_FAILURE_MODULE = textwrap.dedent(
    """
    class DoublerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.Predict("value: int -> result: int")

        def forward(self, **inputs):
            r = self.solve(value=inputs["value"])
            return dspy.Prediction(result=r.result)
    """
).strip()

RECOVERED_LM_FAILURE_MODULE = textwrap.dedent(
    """
    class DoublerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.Predict("value: int -> result: int")

        def forward(self, **inputs):
            try:
                r = self.solve(value=inputs["value"])
            except dspy.LMRateLimitError:
                pass  # recover from the provider failure, caught by its type...
            return oops  # ...then crash locally (NameError)
    """
).strip()


@pytest.mark.parametrize("factory", BACKENDS)
def test_lm_failure_keeps_its_type_across_the_boundary(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(LM_FAILURE_MODULE)
    with dspy.context(lm=_rate_limited_lm()), pytest.raises(LMRateLimitError) as err:
        flex(value=2)
    assert isinstance(err.value, LMError)  # catchable by the family callers actually handle
    assert isinstance(err.value.__cause__, CodeExecutionError)  # the sandbox error stays attached


@pytest.mark.parametrize("factory", BACKENDS)
def test_recovered_lm_failure_does_not_hijack_a_later_crash(factory) -> None:
    flex = Flex(Doubler, interpreter_factory=factory)
    flex._bind_code(RECOVERED_LM_FAILURE_MODULE)
    with dspy.context(lm=_rate_limited_lm()), pytest.raises(CodeInterpreterError) as err:
        flex(value=2)
    assert not isinstance(err.value, LMError)  # the local NameError is not repainted as infra
    assert "NameError" in str(err.value) or "oops" in str(err.value)


# =============================================================================
# Integration (Deno) — the real run-in-interpreter bridge
# =============================================================================

# An optimized CoT module with optimizer-authored glue: the ChainOfThought call must bridge to the
# host, while the import/regex/int parsing run in the sandbox.
COT_GLUE_MODULE = textwrap.dedent(
    """
    class DoublerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("value: int -> result: str")

        def forward(self, **inputs):
            r = self.solve(value=inputs["value"])
            try:
                answer = int(r.result)
            except (ValueError, TypeError):
                import re
                m = re.search(r"-?\\d+", str(r.result))
                answer = int(m.group()) if m else 0
            return dspy.Prediction(result=answer)
    """
).strip()


@deno_required
def test_optimized_cot_runs_in_sandbox_with_lm_bridged() -> None:
    dspy.configure(lm=DummyLM([{"plan": "double of 21 is 42", "result": "the result is 42"}]))
    flex = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
    flex._bind_code(COT_GLUE_MODULE)
    # forward() runs entirely in the sandbox; self.solve(...) bridges to the host DummyLM, and the
    # `import re` / int-parsing glue executes inside Pyodide.
    out = flex(value=21)
    assert isinstance(out, dspy.Prediction)
    assert out.result == 42
    # The predictor lived in the forward's invocation, not on the module: nothing is attached,
    # so parallel forwards on one Flex cannot overwrite each other's predictors.
    assert not hasattr(flex, "solve")


# Adversarial module: tries to read a host file. In the sandbox this hits Pyodide's virtual FS, which
# has no such file, so it is contained — the host file is never exposed.
def _evil_read_module(host_path: str) -> str:
    return textwrap.dedent(
        f"""
        class DoublerModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                try:
                    with open({host_path!r}) as f:
                        leaked = f.read()
                except Exception as e:
                    leaked = "BLOCKED:" + type(e).__name__
                return dspy.Prediction(result=leaked)
        """
    ).strip()


@deno_required
def test_forward_runs_on_a_fresh_instance_each_call() -> None:
    """Each forward instantiates the module anew in the sandbox — instance state does not carry
    across calls. This is deliberate: GEPA scores every rollout under these semantics, so
    deployment must match; it also keeps scores independent of evaluation order and of which
    interpreter serves the call. The primitives catalog tells the proposer as much.
    """
    counter_src = textwrap.dedent(
        """
        class DoublerModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                return dspy.Prediction(result=self.calls)
        """
    ).strip()

    program = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
    program._bind_code(counter_src)
    assert program(value=1).result == 1
    assert program(value=1).result == 1  # not 2: a persistent instance would diverge from scoring


@deno_required
def test_bridged_program_survives_save_load(tmp_path) -> None:
    dspy.configure(lm=DummyLM([{"plan": "double of 21 is 42", "result": "the result is 42"}]))
    program = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
    path = tmp_path / "prog.json"
    program._bind_code(COT_GLUE_MODULE)
    program.save(path)

    # The interpreter is a runtime dependency (like the LM): the saved code persists, the sandbox does not.
    assert "interpreter" not in path.read_text()

    # Reconstruct with a fresh interpreter factory and load the saved code.
    reloaded = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
    reloaded.load(path)
    assert reloaded.module_src and "self.solve" in reloaded.module_src
    out = reloaded(value=21)  # still runs in the sandbox, LM bridged to the host
    assert out.result == 42


# A module whose glue calls a user-provided tool directly (no LM): proves Flex tools are callable
# from inside the sandbox via the host bridge.
TOOL_GLUE_MODULE = textwrap.dedent(
    """
    class ShoutModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            return dspy.Prediction(out=shout(inputs["text"]))
    """
).strip()


@deno_required
def test_user_tool_is_callable_from_sandbox_glue() -> None:
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter())
    flex._bind_code(TOOL_GLUE_MODULE)
    # shout() runs on the host (bridged from the sandbox); no LM involved.
    assert flex(text="hello").out == "HELLO"


RUNAWAY_MODULE = textwrap.dedent(
    """
    class DoublerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("value: int -> result: str")

        def forward(self, **inputs):
            n = 0
            for _ in range(100):           # runaway glue: would drive 100 host LM calls
                self.solve(value=inputs["value"])
                n += 1
            return dspy.Prediction(result=n)
    """
).strip()


@deno_required
def test_runaway_glue_is_stopped_by_budget_in_sandbox() -> None:
    dspy.configure(lm=DummyLM([{"plan": "r", "result": "1"} for _ in range(50)]))
    flex = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter(), max_predictor_calls=3)
    flex._bind_code(RUNAWAY_MODULE)
    with pytest.raises(Exception) as excinfo:  # the budget error propagates out of the sandbox
        flex(value=1)
    assert "budget" in str(excinfo.value).lower()


# A module that constructs and calls a dspy.ReAct with a Flex-level tool. ReAct runs no code of its
# own: the LM picks a tool, the host runs it, and the LM submits the answer; the agent itself runs in
# the sandbox, and only the tool call and the LM calls cross to the host.
REACT_MODULE = textwrap.dedent(
    """
    class ShoutModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.agent = dspy.ReAct("text: str -> out: str", tools=[shout])

        def forward(self, **inputs):
            r = self.agent(text=inputs["text"])
            return dspy.Prediction(out=r.out)
    """
).strip()


@deno_required
def test_bridged_react_runs_in_flex_sandbox() -> None:
    dspy.configure(
        lm=DummyLM(
            [
                {
                    "next_thought": "shout it",
                    "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "shout", "args": {"text": "hello"}}]),
                },
                {
                    "next_thought": "the tool returned HELLO",
                    "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"out": "HELLO"}}]),
                },
            ]
        )
    )
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter())
    flex._bind_code(REACT_MODULE)
    out = flex(text="hello")
    # The agent's glue ran in the sandbox; the `shout` tool call and every LM call bridged to the host.
    assert out.out == "HELLO"


@deno_required
def test_adversarial_host_file_access_is_contained(tmp_path) -> None:
    secret = tmp_path / "host_secret.txt"
    secret.write_text("TOP-SECRET")
    assert secret.read_text() == "TOP-SECRET"  # readable in-process on the host

    class Leak(dspy.Signature):
        value: int = dspy.InputField()
        result: str = dspy.OutputField()

    flex = Flex(Leak, interpreter_factory=lambda: dspy.PythonInterpreter())
    flex._bind_code(_evil_read_module(str(secret)))
    out = flex(value=1)
    # The sandbox could not reach the host file; the secret never crossed the boundary.
    assert "TOP-SECRET" not in str(out.result)
    assert str(out.result).startswith("BLOCKED")


@deno_required
def test_sandboxed_code_reads_no_host_secret_and_has_the_host_import_nothing(tmp_path, monkeypatch) -> None:
    # Signature strings naming host objects or modules, built in the sandbox and recorded to the host's trace; and the
    # host LM's settings, credentials among them, which requests built for the sandbox carry.
    (tmp_path / "planted_module.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DSPY_TEST_HOST_SECRET", "env-host-secret")

    class Leak(dspy.Signature):
        value: int = dspy.InputField()
        result: str = dspy.OutputField()

    flex = Flex(Leak, interpreter_factory=dspy.PythonInterpreter)
    flex._bind_code(
        "class M(dspy.Module):\n"
        "    def forward(self, value):\n"
        "        seen = []\n"
        "        for annotation in ('os.environ', 'dspy.settings.lm.kwargs', 'Literal[os.environ]', 'planted_module'):\n"
        "            signature = 'x: ' + annotation + ' -> y'\n"
        "            try:\n"
        "                seen.append(repr(dspy.Predict(signature).signature.fields))\n"
        "            except Exception as e:\n"
        "                seen.append(str(e))\n"
        "            entry = {'signature': {'signature': signature, 'types': {}, 'inputs': ['x'], 'outputs': ['y']},\n"
        "                     'inputs': {'x': '1'}}\n"
        "            seen.append(repr(dspy.settings.trace._host(entry=entry)))\n"
        "        seen.append(repr(dspy.settings.lm._build_adapter_request('hi', {})))\n"
        "        return dspy.Prediction(result='\\n'.join(seen))\n"
    )
    lm = dspy.LM("openai/gpt-4o-mini", api_key="sk-host-secret", aws_secret_access_key="aws-host-secret")
    with dspy.context(lm=lm, trace=[]):
        out = flex(value=1).result  # no LM call: building a request does not reach the provider
        trace = dspy.settings.trace
    assert "host-secret" not in out
    assert "planted_module" not in sys.modules
    assert len(trace) == 4 and all(predictor.signature.fields["x"].annotation is str for predictor, *_ in trace)


@deno_required
def test_nested_rlm_tool_provenance() -> None:
    """A GEPA-authored RLM spawns a second RLM through a host tool, and a host-provided tool is
    threaded down to both. Assert the trust boundary AND that each layer is a distinct interpreter:

      - each layer runs in a Pyodide sandbox (its ``sys.platform`` is 'emscripten', not the host's);
      - each layer is its OWN interpreter, not one shared sandbox. A global counter persists WITHIN
        an interpreter but resets in a fresh one, so the Flex glue counts 1 then 2 in interp A, while
        each nested agent sees 1 (a fresh namespace) — if they shared the glue's interpreter it would
        read 3, then 4;
      - the host-provided ``probe`` tool, wherever it is called from, executes in the HOST process.

    A code-executing agent's REPL has no ``dspy`` shim, so the second RLM is spawned via a host tool
    (``spawn_inner``) that constructs and runs it on the host, passing ``probe`` further down; only
    that sub-agent's ``out`` string crosses back, so its trajectory never touches the Flex bridge.
    """
    host_platform = sys.platform
    assert host_platform != "emscripten"  # sanity: the test itself is the real host, not the sandbox
    record: dict[str, dict[str, str]] = {}
    host_seen: dict[str, str] = {}

    def probe(layer: str, platform: str, count: str) -> str:
        # Host-provided tool: runs in the host process. Records, per layer, the platform and the
        # per-interpreter counter value that layer reported, plus the platform `probe` itself runs in.
        record[layer] = {"platform": platform, "count": count}
        host_seen["platform"] = __import__("sys").platform
        return "ok"

    def spawn_inner(task: str) -> str:
        # Host tool that spawns a SECOND code-executing agent (a nested interpreter), passing the
        # host `probe` tool down to it. Runs on the host; only the inner RLM's `out` string is
        # returned, so nothing but a string crosses back.
        inner = dspy.RLM("task: str -> out: str", tools=[probe])
        return inner(task=task).out

    class ProbeSig(dspy.Signature):
        task: str = dspy.InputField()
        out: str = dspy.OutputField()

    # Each agent bumps a global counter and reports it; a fresh interpreter starts the count at 1.
    outer_code = (
        "```python\nimport sys\ng = globals()\ng['_n'] = g.get('_n', 0) + 1\n"
        "probe(layer='outer_rlm', platform=sys.platform, count=str(g['_n']))\n"
        "spawn_inner(task='go')\nSUBMIT(out='outer')\n```"
    )
    inner_code = (
        "```python\nimport sys\ng = globals()\ng['_n'] = g.get('_n', 0) + 1\n"
        "probe(layer='inner_rlm', platform=sys.platform, count=str(g['_n']))\nSUBMIT(out='inner')\n```"
    )
    dspy.configure(
        lm=DummyLM(
            [
                {"plan": "probe here, then spawn the sub-agent", "code": outer_code},
                {"plan": "probe from the sub-agent", "code": inner_code},
            ]
        )
    )

    module = textwrap.dedent(
        """
        class ProbeModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.rlm = dspy.RLM("task: str -> out: str", tools=[probe, spawn_inner])

            def forward(self, **inputs):
                import sys
                g = globals()
                g["_n"] = g.get("_n", 0) + 1                                        # interp A -> 1
                probe(layer="glue_1", platform=sys.platform, count=str(g["_n"]))
                g["_n"] = g.get("_n", 0) + 1                                        # same interp -> 2
                probe(layer="glue_2", platform=sys.platform, count=str(g["_n"]))
                self.rlm(task=inputs["task"])
                return dspy.Prediction(out="done")
        """
    ).strip()

    flex = Flex(ProbeSig, tools=[probe, spawn_inner], interpreter_factory=lambda: dspy.PythonInterpreter())
    flex._bind_code(module)
    flex(task="go")

    # Every layer ran inside a Pyodide sandbox, not the host process:
    for layer in ("glue_1", "glue_2", "outer_rlm", "inner_rlm"):
        assert record[layer]["platform"] == "emscripten", f"Failed at {layer}."
    # The counter persists WITHIN the Flex glue's interpreter (1 then 2)...
    assert record["glue_1"]["count"] == "1"
    assert record["glue_2"]["count"] == "2"
    # ...but each nested agent starts fresh at 1 — proving it is its OWN interpreter, not the glue's
    # (which would read 3, then 4) and not each other's:
    assert record["outer_rlm"]["count"] == "1"
    assert record["inner_rlm"]["count"] == "1"
    # The host-provided tool itself executed in the host process, wherever it was called from:
    assert host_seen["platform"] == host_platform
