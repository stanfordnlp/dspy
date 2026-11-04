"""SandboxLM: the host LM as sandboxed code may call it, and the wire between a sandbox's HostLM and the host."""

import builtins
import copy
import dataclasses
import datetime
import enum
import inspect
import json
import pickle
import sys
import types
import typing

import pydantic
import pydantic_core
import pytest

import dspy
from dspy.adapters.base import Prompt
from dspy.lm15 import CacheConfig, Config, FunctionTool, Message, Request, Response
from dspy.primitives.facade import LM_ERROR_KEY, SANDBOX_LM_KWARGS, SandboxLM, SandboxSession, from_wire, types_spec
from dspy.primitives.sandbox_guest import (
    HostLM,
    decode_options,
    decode_prompt,
    encode_options,
    encode_prompt,
    encode_value,
    rebuild_types,
)
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMProviderError, LMRateLimitError


def _host_lm(session=None):
    """The sandbox's view of the host LM, wired straight to a session in this process."""
    session = session or SandboxSession({})
    return HostLM(session._serve_lm, session._spec("test")["lm"])


class _CapableLM(DummyLM):
    supports_function_calling = True
    supports_response_schema = True
    supported_params = frozenset({"temperature", "response_format", "tools"})


def _lm():
    return DummyLM([{"answer": "a"}, {"answer": "b"}])


def test_generation_parameters_pass_and_the_wrapped_lm_does_the_work():
    lm = _lm()
    request = Request(model=lm.model, messages=(Message.user("hi"),), config=Config(temperature=0.3, max_tokens=5))
    output = SandboxLM(lm)(request)
    assert isinstance(output, Response)
    assert output.message.text == "<answer>\na\n</answer>"
    assert len(lm.history) == 1
    assert lm.history[0]["request"].config == request.config


@pytest.mark.parametrize(
    "options",
    [{"api_base": "http://evil.example"}, {"api_key": "sk-x"}, {"extra_headers": {"X": "y"}}, {"model": "gpt-5-pro"}],
)
def test_routing_credential_and_transport_options_are_rejected_before_any_request(options):
    lm = _lm()
    sandbox = SandboxLM(lm)
    with pytest.raises(TypeError, match="may not set LM option"):
        sandbox("prompt", temperature=0.1, **options)
    with pytest.raises(TypeError, match="may not set LM option"):
        sandbox.copy(**options)
    assert lm.history == []
    assert not options.keys() & SANDBOX_LM_KWARGS


def test_typed_requests_keep_the_settings_the_host_lm_owns():
    import dataclasses

    lm = dspy.LM("openai/gpt-4o-mini", frequency_penalty=0.1, extra_body={"a": 1})
    sandbox = SandboxLM(lm)
    request = sandbox._build_adapter_request("hi", {"temperature": 0.5})
    sandbox._check_request(dataclasses.replace(request, config=dataclasses.replace(request.config, seed=3)))
    for change in ({"extensions": {"custom_llm_provider": "anthropic"}}, {"frequency_penalty": None}):
        with pytest.raises(TypeError, match=rf"may not change request setting\(s\) \[{next(iter(change))!r}\]"):
            sandbox(dataclasses.replace(request, config=dataclasses.replace(request.config, **change)))


def test_typed_requests_the_host_built_keep_how_it_normalized_its_settings():
    class Recording(dspy.LM):
        def __call__(self, *args, **kwargs):
            calls.append(args)
            return "ok"

    calls = []
    # An effort override moves this LM's reasoning_format out of the request's extensions.
    session = SandboxSession({}, lm=Recording("groq/openai/gpt-oss-20b", reasoning_format="parsed"))
    built = session._serve_lm("request", prompt="hi", options={"reasoning_effort": "low"})
    assert session._serve_lm("call", prompt={"request": built}) == "ok"
    assert len(calls) == 1


def test_requests_built_for_the_sandbox_leave_the_settings_the_host_lm_owns_on_the_host():
    class Recording(dspy.LM):
        def __call__(self, request, **kwargs):
            calls.append(request)
            return "ok"

    calls = []
    # Provider options, which may be credentials (here for a provider LiteLLM serves), ride in a request's extensions.
    lm = Recording("openai/gpt-4o-mini", aws_secret_access_key="aws-secret", extra_body={"token": "body-secret"},
                   user="alice", frequency_penalty=0.1)
    session = SandboxSession({}, lm=lm)
    built = session._serve_lm("request", prompt="hi", options={"temperature": 0.5})
    wire = json.dumps(built)
    assert "secret" not in wire and "alice" not in wire and "frequency_penalty" not in wire
    assert built["config"]["temperature"] == 0.5

    assert session._serve_lm("call", prompt={"request": built}) == "ok"
    config = calls[0].config
    assert config.extensions == {"aws_secret_access_key": "aws-secret", "extra_body": {"token": "body-secret"}}
    assert (config.user_id, config.frequency_penalty, config.temperature) == ("alice", 0.1, 0.5)

    for change in (
        {"extensions": {**built["config"]["extensions"], "api_base": "http://evil.example"}},
        {"extensions": {"__dspy_host_settings__": 7}},
        {"extensions": None},
        {"user_id": "mallory"},
    ):
        tampered = {**built, "config": {**built["config"], **change}}
        with pytest.raises(TypeError, match="may not change request setting"):
            session._serve_lm("call", prompt={"request": tampered})
    assert len(calls) == 1


def test_settings_of_a_copied_host_lm_reach_each_call():
    seen = []

    class Recording(DummyLM):
        def __call__(self, prompt=None, cache=None, rollout_id=None, **options):
            seen.append((cache, rollout_id, options.get("temperature")))
            return super().__call__(prompt, cache=cache, rollout_id=rollout_id, **options)

    lm = _host_lm(SandboxSession({}, lm=Recording([{"answer": "a"}] * 3)))
    copied = lm.copy(rollout_id=2, cache=False, temperature=0.7)  # what BestOfN and Refine do per attempt
    copied(copied._build_adapter_request("hi", {}))  # a typed request already holds the temperature
    copied("hi")
    lm("hi")
    assert seen == [(False, 2, None), (False, 2, 0.7), (None, None, None)]


@pytest.mark.asyncio
async def test_acall_applies_the_same_allowlist():
    lm = _lm()
    request = Request(model=lm.model, messages=(Message.user("prompt"),), config=Config(temperature=0.2))
    with pytest.raises(TypeError, match=r"may not set LM option\(s\) \['api_key'\]"):
        await SandboxLM(lm).acall(request, api_key="sk-x")
    assert lm.history == []
    output = await SandboxLM(lm).acall(request)
    assert isinstance(output, Response)
    assert output.message.text == "<answer>\na\n</answer>"


def test_reserve_meters_admitted_calls_only():
    reserved = []
    lm = _lm()
    lm.kwargs["n"] = 4  # Typed requests remain singular despite a legacy sampling default.
    sandbox = SandboxLM(lm, reserve=reserved.append)

    def request(text):
        return Request(model=lm.model, messages=(Message.user(text),))

    sandbox(request("one"))
    with pytest.raises(TypeError):
        sandbox(request("two"), api_key="sk-x")
    sandbox.copy(temperature=0.9)(request("three"))
    sandbox(request=request("four"))
    with pytest.raises(TypeError):
        sandbox(request("five"), api_base="http://evil.example")
    assert reserved == [1, 1, 1]


def test_reserve_charges_one_slot_per_requested_completion():
    # `n` asks for n completions in one request; it must not multiply max_llm_calls.
    class _StubLM:
        model = "stub"

        def __init__(self):
            self.kwargs = {"n": 4, "api_key": "sk-host"}

        def __call__(self, *args, **kwargs):
            return ["x"] * kwargs.get("n", self.kwargs["n"])

    reserved = []
    sandbox = SandboxLM(_StubLM(), reserve=reserved.append)
    assert len(sandbox("per-call", n=64)) == 64
    sandbox("mirrored default")
    sandbox("explicit one", n=1)
    assert reserved == [64, 4, 1]


def test_proxy_mirrors_identity_kwargs_and_capabilities_of_the_wrapped_lm():
    lm = _CapableLM([{"answer": "a"}])
    lm.kwargs.update(temperature=0.7, api_key="sk-secret", api_base="http://internal.example")
    sandbox = SandboxLM(lm)
    assert (sandbox.model, sandbox.model_type) == (lm.model, lm.model_type)
    assert sandbox.kwargs["temperature"] == 0.7
    assert not {"api_key", "api_base"} & sandbox.kwargs.keys()  # credentials stay on the wrapped LM
    assert sandbox.supports_function_calling and sandbox.supports_response_schema and not sandbox.supports_reasoning
    assert sandbox.supported_params == {"temperature", "response_format", "tools"}
    assert not SandboxLM(_lm()).supports_function_calling


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "bridge", "two-step-async"])
async def test_predictions_keep_trusted_host_defaults(mode):
    lm = _lm()
    lm.kwargs.update(frequency_penalty=0.6, top_k=7, extra_body={"host_option": "enabled"}, temperature=0.2)
    reserved = []
    sandbox = SandboxLM(SandboxLM(lm), reserve=reserved.append)
    adapter = dspy.ChatAdapter()
    if mode == "two-step-async":
        extraction_lm = _lm()
        lm.adapter = extraction_lm.adapter = adapter
        adapter = dspy.TwoStepAdapter(extraction_lm)
    with dspy.context(lm=sandbox, adapter=adapter):
        if mode == "bridge":
            assert dspy.Predict("question -> answer", temperature=0.7)(question="hello", lm=_host_lm()).answer == "a"
        else:
            predictor = dspy.Predict("question -> answer", temperature=0.7)
            result = predictor(question="hello") if mode == "sync" else await predictor.acall(question="hello")
            assert result.answer == "a"
    [entry] = lm.history
    config = entry["request"].config
    assert config.frequency_penalty == 0.6
    assert config.top_k == 7
    assert config.extensions["extra_body"] == {"host_option": "enabled"}
    assert config.temperature == 0.7
    assert lm.kwargs["temperature"] == 0.2
    assert not {"frequency_penalty", "extra_body"} & sandbox.kwargs.keys()
    assert reserved == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "bridge", "two-step-async"])
@pytest.mark.parametrize("options", [
    {"frequency_penalty": 0.6}, {"extra_body": {"host_option": "enabled"}},
    {"extra_body": None}, {"api_key": None}, {"num_generations": 2},
])
async def test_predict_options_cannot_bypass_sandbox_allowlist(mode, options):
    lm = _lm()
    lm.kwargs.update(frequency_penalty=0.6, extra_body={"host_option": "enabled"})
    reserved = []
    sandbox = SandboxLM(lm, reserve=reserved.append)
    adapter = dspy.TwoStepAdapter(_lm()) if mode == "two-step-async" else dspy.ChatAdapter()
    with dspy.context(lm=sandbox, adapter=adapter), pytest.raises(TypeError, match="may not set LM option"):
        if mode == "bridge":
            dspy.Predict("question -> answer")(question="hello", config=options, lm=_host_lm())
        else:
            predictor = dspy.Predict("question -> answer")
            if mode == "sync":
                predictor(question="hello", config=options)
            else:
                await predictor.acall(question="hello", config=options)
    assert lm.history == []
    assert reserved == []


def test_wraps_a_callable_only_lm():
    # RLM accepts any callable as sub_lm; the facade is installed by default, so the proxy must too.
    class PlainLM:
        def __call__(self, prompt):
            return ["plain:" + prompt]

    sandbox = SandboxLM(PlainLM())
    assert sandbox("hi") == ["plain:hi"]
    assert (sandbox.model, sandbox.kwargs, sandbox.supports_function_calling, sandbox.supported_params) == (
        None, {}, False, set()
    )


def test_predict_wraps_a_request_callable_without_lm_metadata():
    class Host:
        model = "custom/host"

        def __call__(self, request):
            assert request.config.temperature == 0.3
            return Response(None, self.model, Message.assistant("<answer>a</answer>"), "stop", dspy.lm15.Usage())

    result = dspy.Predict("question -> answer", temperature=0.3)(question="hello", lm=SandboxLM(Host()))
    assert result.answer == "a"


def test_host_lm_mirrors_the_host_lm_without_its_credentials():
    lm = _CapableLM([{"answer": "a"}])
    lm.kwargs.update(temperature=0.7, api_key="sk-secret", extra_body={"host_option": "enabled"})
    with dspy.context(lm=lm):
        host = _host_lm()
    assert (host.model, host.model_type) == (lm.model, lm.model_type)
    assert host.kwargs["temperature"] == 0.7
    assert not {"api_key", "extra_body"} & host.kwargs.keys()
    assert host.supports_function_calling and host.supports_response_schema and not host.supports_reasoning
    assert host.supported_params == {"temperature", "response_format", "tools"}


def test_host_lm_copy_overrides_like_call_options():
    lm = _lm()
    with dspy.context(lm=lm):
        host = _host_lm()
        assert dspy.Predict("question -> answer")(question="hi", lm=host.copy(temperature=0.9)).answer == "a"
        with pytest.raises(TypeError, match=r"may not set LM option\(s\) \['frequency_penalty'\]"):
            dspy.Predict("question -> answer")(question="hi", lm=host.copy(frequency_penalty=0.5))
    assert lm.history[0]["request"].config.temperature == 0.9


def test_string_prompts_cross_and_are_metered_per_completion():
    reserved = []
    lm = DummyLM([{"answer": "a"}, {"answer": "b"}])
    with dspy.context(lm=lm):
        host = _host_lm(SandboxSession({}, reserve=reserved.append))
        assert len(host("hello", n=2)) == 2
    assert reserved == [2]


def test_lm_errors_cross_as_their_type_with_a_tag_the_host_recognizes():
    from tests.test_utils.engines import failing_lm

    session = SandboxSession({})
    with dspy.context(lm=failing_lm(LMRateLimitError("429", model="dummy"))):
        host = _host_lm(session)
        with pytest.raises(LMRateLimitError, match=r"\[dspy bridge lm-error #\d+\]") as err:
            dspy.Predict("question -> answer")(question="hi", lm=host)
    original, tag = session.lm_error
    assert isinstance(original, LMRateLimitError) and tag in str(err.value)


def test_lm_errors_reach_the_sandbox_without_the_host_lm_credentials():
    from tests.test_utils.engines import failing_lm

    secrets_ = {"api_key": "sk-CANARY-key-0123", "aws_secret_access_key": "aws-CANARY-secret-4567"}
    lm = failing_lm(LMProviderError(f"401: bad key {secrets_['api_key']} for {secrets_['aws_secret_access_key']}",
                                    model="dummy"))
    lm.kwargs.update(secrets_)
    session = SandboxSession({})
    with dspy.context(lm=lm):
        message = session._serve_lm("call", prompt="hi")[LM_ERROR_KEY]["message"]
    assert "401: bad key *** for ***" in message and "CANARY" not in message
    assert "CANARY" in str(session.lm_error[0])  # the host keeps the original error


def test_prompts_and_options_round_trip_the_wire():
    class Answer(pydantic.BaseModel):
        text: str

    prompt = Prompt(messages=(Message.user("hi"),), system="be brief")
    request = Request(model="m", messages=(Message.user("hi"),), config=Config(temperature=0.1))
    assert decode_prompt(encode_prompt(prompt)) == prompt
    assert decode_prompt(encode_prompt(request)) == request
    assert decode_prompt(encode_prompt("hi")) == "hi"

    tool = FunctionTool(name="f", description="d", parameters={"type": "object", "properties": {}})
    options = encode_options({"response_format": Answer, "tools": [tool], "prompt_cache": CacheConfig(), "n": 2})
    import json

    decoded = decode_options(json.loads(json.dumps(options)))
    assert decoded["response_format"]["json_schema"]["name"] == "Answer"
    assert decoded["tools"] == [tool]
    assert decoded["prompt_cache"] == CacheConfig()
    assert decoded["n"] == 2


class _Node(pydantic.BaseModel):
    value: int
    children: list["_Node"] = []


def test_recursive_models_keep_their_fields_in_the_sandbox():
    rebuilt = rebuild_types(types_spec({"_Node": _Node}))["_Node"]
    tree = {"value": 1, "children": [{"value": 2, "children": [{"value": 3}]}]}
    assert rebuilt.model_validate(tree).model_dump(mode="json") == _Node.model_validate(tree).model_dump(mode="json")


@pytest.mark.parametrize(
    "adapter",
    [
        dspy.XMLAdapter(use_native_function_calling=True, parallel_tool_calls=False),
        dspy.ChatAdapter(use_json_adapter_fallback=False, native_response_types=[dspy.Reasoning]),
        dspy.JSONAdapter(use_native_function_calling=False),
    ],
)
def test_built_in_adapters_keep_their_settings_in_the_sandbox(adapter):
    import json

    from dspy.primitives.facade import _adapter_spec
    from dspy.primitives.sandbox_guest import rebuild_adapter

    spec, _ = _adapter_spec(adapter)
    rebuilt = rebuild_adapter(json.loads(json.dumps(spec)))
    assert type(rebuilt) is type(adapter)
    for name in ("use_native_function_calling", "parallel_tool_calls", "use_json_adapter_fallback", "native_response_types"):
        assert getattr(rebuilt, name, None) == getattr(adapter, name, None)


def test_extended_trace_entries_reach_the_host_too():
    from dspy.primitives.sandbox_guest import HostTrace

    sent = []
    trace = HostTrace(lambda entry: sent.append(entry))
    entry = (dspy.Predict("question -> answer"), {"question": "q"}, dspy.Prediction(answer="a"))
    trace.append(entry)
    trace.extend([entry, entry])  # how BestOfN and Refine add their chosen attempt
    trace += [entry]
    assert len(trace) == len(sent) == 4
    assert sent[0]["outputs"] == {"answer": "a"}


def test_signatures_the_sandbox_records_import_nothing_and_name_no_object_on_the_host(tmp_path, monkeypatch):
    # A module the host could import, and a standard-library module it has not imported.
    (tmp_path / "planted_module.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "tabnanny", raising=False)
    session = SandboxSession({})

    def recorded(signature, types=None):
        session._serve_trace({
            "signature": {"signature": signature, "instructions": "i", "types": types or {},
                          "inputs": ["x"], "outputs": ["y"]},
            "inputs": {"x": "1"}, "outputs": {"y": "2"},
        })
        return {name: field.annotation for name, field in dspy.settings.trace[-1][0].signature.fields.items()}

    with dspy.context(trace=[]):
        assert recorded("x: planted_module -> y") == {"x": str, "y": str}
        assert recorded("x: T -> y", {"T": "tabnanny.NannyNag"}) == {"x": str, "y": str}
        for reach in ("os.environ", "dspy.settings.lm.kwargs", "Literal[os.environ]", "Literal[dspy.settings.lm]",
                      "Literal[sys]", "list[os.sep]"):
            assert recorded(f"x: {reach} -> y") == {"x": str, "y": str}, reach
        # A quoted type is a forward reference, which pydantic would evaluate as Python on the host.
        for quoted in ("list[\"(__import__('builtins').__setattr__('PWNED', 1), int)[1]\"]",
                       "\"__import__('builtins').__setattr__('PWNED', 1)\"", "dict[str, 'PWNED']",
                       "Literal[\"x\"] | 'int'"):
            assert recorded(f"x: {quoted} -> y") == {"x": str, "y": str}, quoted
        assert not hasattr(builtins, "PWNED")
        # Nor does a class the sandbox calls Literal take quoted values.
        assert recorded("x: Literal['int'] -> y", {"Literal": "typing.Generic"}) == {"x": str, "y": str}
        # What the sandbox records for its predictors still resolves: builtin, typing, dspy and loaded stdlib types.
        types = {"Reasoning": "dspy.adapters.types.reasoning.Reasoning", "date": "datetime.date"}
        assert recorded("x: list[int] | None, when: date -> y: Reasoning, z: Literal['a', -1]", types) == {
            "x": list[int] | None, "when": datetime.date, "y": dspy.Reasoning, "z": typing.Literal["a", -1],
        }
    assert "planted_module" not in sys.modules and "tabnanny" not in sys.modules


def test_types_the_sandbox_names_are_read_without_running_a_module_getattr(monkeypatch):
    from dspy.primitives import facade

    called = []
    lazy = types.ModuleType("lazymod")
    lazy.__getattr__ = lambda name: called.append(name) or int  # a module that loads attributes lazily
    monkeypatch.setitem(sys.modules, "lazymod", lazy)
    monkeypatch.setattr(facade, "_in_sandbox", lambda module: True)
    with pytest.raises(ValueError):
        facade._loaded_class("lazymod.Thing")
    assert called == []


def test_stand_ins_for_sandbox_functions_take_their_parameter_kinds_by_name():
    from dspy.primitives.facade import _NestedInterpreter

    nested = _NestedInterpreter(None, {})
    stand_in = nested._parent_call("helper", {"parameters": [{"name": "a", "kind": "KEYWORD_ONLY", "type": "int"}]})
    assert str(inspect.signature(stand_in)) == "(*, a: int)"
    with pytest.raises(KeyError) as error:  # not an attribute of a host class, shown back to the sandbox
        nested._parent_call("helper", {"parameters": [{"name": "a", "kind": "__dict__"}]})
    assert str(error.value) == "'__dict__'"


# -- custom types, which cross as their shape, and values, which cross by field name ---------------------------------


class _Hue(str, enum.Enum):
    RED = "red"
    CRIMSON = "red"  # an alias


class _Mark(enum.IntFlag):
    R = 4
    W = 2


@dataclasses.dataclass(frozen=True, order=True)
class _Spot:
    x: int
    label: str = dataclasses.field(default="", compare=False)


class _Pair(typing.NamedTuple):
    left: int
    right: str = "r"


class _Count(pydantic.RootModel[int]):
    root: int = 3


_API_KEY = "sk-CANARY"  # a private value of the module


def _clean(value: str) -> str:  # a helper a validator calls
    return value.strip().lower()


class _Account(pydantic.BaseModel):
    """An account."""

    model_config = pydantic.ConfigDict(ser_json_bytes="hex", val_json_bytes="hex")
    name: str = pydantic.Field(alias="Name", max_length=20)
    last: int = pydantic.Field(validation_alias=pydantic.AliasPath("items", -1))
    id: str = pydantic.Field(default_factory=lambda: "made")
    hue: _Hue = _Hue.RED
    mark: _Mark = _Mark.R
    spots: set[_Spot] = set()
    pair: _Pair = _Pair(1)
    count: _Count = _Count()
    data: bytes = b""
    when: datetime.datetime | None = None
    token: str = pydantic.Field(default="", exclude=True)
    key: str = pydantic.Field(default=_API_KEY, exclude=True)
    kids: list["_Account"] = []

    @pydantic.field_validator("name")
    @classmethod
    def _name(cls, value):
        return _clean(value)

    def greeting(self):
        return f"Hi {self.name}"


def _cross(t, value):
    """``value`` sent into a sandbox that builds ``t``, and its value there sent back, as the bridge does."""
    inside = from_wire(encode_value(value), rebuild_types(types_spec({t.__name__: t}))[t.__name__])
    return inside, from_wire(encode_value(inside), t)


def test_custom_types_cross_as_their_shape():
    assert "CANARY" not in json.dumps(types_spec({"_Account": _Account}))  # nothing but the shape crosses
    original = _Account.model_validate({
        "Name": " Ann ", "items": [9, 7], "spots": [{"x": 1}], "data": "6869", "mark": 6, "token": "secret",
        "when": "2026-11-02T10:00:00", "kids": [{"Name": "Kid", "items": [1]}],
    })
    inside, back = _cross(_Account, original)
    assert type(inside) is not _Account and (type(inside).__name__, type(inside).__doc__) == ("_Account", "An account.")
    # What the host validated, by field name. Excluded fields never cross, nor does a default that may be private.
    assert (inside.name, inside.last, inside.id, inside.data, inside.token, inside.key) == ("ann", 7, "made", b"hi", "", None)
    assert (inside.hue is type(inside.hue).CRIMSON, int(inside.mark), type(next(iter(inside.spots))).__name__) == (
        True, 6, "_Spot"
    )
    assert (inside.pair.right, inside.count.root, inside.when.year, type(inside.kids[0]).__name__) == (
        "r", 3, 2026, "_Account"
    )
    assert back == original.model_copy(update={"token": ""})


def test_methods_and_validators_stay_on_the_host():
    inside = rebuild_types(types_spec({"_Account": _Account}))["_Account"]
    assert not hasattr(inside, "greeting") and not hasattr(inside, "_name")
    made = inside(Name=" Bob ", items=[1])  # built in the sandbox: no validator of the user's runs there
    assert made.name == " Bob "
    with pytest.raises(pydantic.ValidationError):  # but its constraints do
        inside(Name="x" * 21, items=[1])
    back = from_wire(encode_value(made), _Account)  # the host's own type reads what comes back
    assert (back.name, back.id, back.greeting()) == ("bob", "made", "Hi bob")


def test_types_defined_in_a_function_cross_as_their_shape():
    def make(limit):
        class Point(pydantic.BaseModel):
            x: int

        class Shape(pydantic.BaseModel):
            points: list[Point]
            cap: int = limit  # read while the class is made

            def capped(self):
                return min(len(self.points), limit)

        return Shape

    shape = make(3)
    inside, back = _cross(shape, shape(points=[{"x": 1}]))
    assert (type(inside.points[0]).__name__, inside.cap, back) == ("Point", 3, shape(points=[{"x": 1}]))


@dataclasses.dataclass
class _Ticket:
    title: str
    id: str = dataclasses.field(default_factory=lambda: "made")  # a factory only the host has


class _Raw(pydantic.RootModel[bytes]):
    model_config = pydantic.ConfigDict(ser_json_bytes="hex", val_json_bytes="hex")


class _Level(pydantic.BaseModel):
    hue: typing.Literal[_Hue.RED] = _Hue.RED


class _Note(pydantic.BaseModel):
    text: str | None = pydantic.Field(default_factory=lambda: "made")


@dataclasses.dataclass
class _Draft:
    text: str | None = dataclasses.field(default_factory=lambda: "made")


def test_defaults_made_on_the_host_enum_literals_and_root_settings_keep_their_meaning():
    inside = rebuild_types(types_spec({"_Ticket": _Ticket}))["_Ticket"]
    assert from_wire(encode_value(inside(title="t")), _Ticket) == _Ticket("t", "made")  # the host made the id
    for host in (_Note, _Draft):
        inside = rebuild_types(types_spec({host.__name__: host}))[host.__name__]
        unset, chosen = inside(), inside()
        chosen.text = "chosen"
        assert repr(unset.text) == "<made on the host>"
        assert from_wire(encode_value(unset), host).text == "made"
        assert from_wire(encode_value(chosen), host).text == "chosen"  # set after it was built
        assert from_wire(encode_value(inside(text=None)), host).text is None  # a None given stays
        for copied in (copy.copy(unset), copy.deepcopy(unset), pickle.loads(pickle.dumps(unset))):
            assert from_wire(encode_value(copied), host).text == "made"
    assert from_wire(encode_value(inside(None)), _Draft).text is None
    assert from_wire(encode_value(dataclasses.replace(inside(), text="new")), _Draft).text == "new"
    ticket = rebuild_types(types_spec({"_Ticket": _Ticket}))["_Ticket"](title="t")
    assert from_wire(encode_value(dataclasses.replace(ticket, title="u")), _Ticket) == _Ticket("u", "made")
    level, _ = _cross(_Level, _Level())
    assert level.hue.name == "RED"  # a member of the enum, not its value
    raw, back = _cross(_Raw, _Raw(b"hi"))
    assert (raw.root, back) == (b"hi", _Raw(b"hi"))


def test_the_host_builds_no_class_from_what_the_sandbox_sends():
    import builtins

    from dspy.primitives.facade import SandboxSession

    escape = {"shape": "(__import__('builtins').__setattr__('_dspy_escaped', 1), int)[1]"}
    sent = {"signature": "case: _Child, note: Verdict -> verdict: list[Verdict], score: int", "instructions": "Judge.",
            "fields": {}, "types": {"Verdict": "__main__.Verdict"}, "inputs": ["case", "note"],
            "outputs": ["verdict", "score"], "shapes": {"types": {"Verdict": escape}, "shapes": {}}}
    signature = SandboxSession({}, custom_types={"_Child": _Child})._signature(sent)
    assert {name: f.annotation for name, f in signature.fields.items()} == {
        "case": _Child, "note": str, "verdict": str, "score": int,  # only a type the host doesn't know is left out
    }
    assert not hasattr(builtins, "_dspy_escaped")


def test_types_built_at_runtime_cross_as_their_shape_too():
    built = pydantic.create_model("Dynamic", a=(int, ...), b=(str, "x"))
    inside = rebuild_types(types_spec({"Dynamic": built}))["Dynamic"]
    assert (inside(a="1").a, inside(a=1).b) == (1, "x")


class _Child(pydantic.BaseModel):
    x: int


class _Parent(pydantic.BaseModel):
    children: "list[_Child]"  # quoted, as pydantic resolves it
    best: "dict[str, list['_Child']]" = {}
    label: typing.Annotated[str, pydantic.StringConstraints(max_length=5)] = "plot"


def test_types_cross_as_pydantic_resolved_them():
    inside, back = _cross(_Parent, _Parent(children=[{"x": 1}], best={"a": [{"x": 2}]}))
    assert (type(inside.children[0]).__name__, inside.best["a"][0].x, back) == (
        "_Child", 2, _Parent(children=[{"x": 1}], best={"a": [{"x": 2}]})
    )
    with pytest.raises(pydantic.ValidationError, match="at most 5 characters"):
        type(inside)(children=[], label="too long")


class _Lamp(pydantic.BaseModel):
    hue: typing.Literal[_Hue.RED] = _Hue.RED
    child: _Child = _Child(x=1)


class _Shade(pydantic.BaseModel):
    child: _Child = _Child(x=2)


def test_a_type_held_by_several_signature_types_is_built_once():
    out = rebuild_types(types_spec({"_Account": _Account, "_Hue": _Hue}))
    assert out["_Account"].model_validate({"Name": "a", "items": [1]}).hue is out["_Hue"].RED
    out = rebuild_types(types_spec({"_Level": _Level, "_Lamp": _Lamp, "_Shade": _Shade}))  # each names one of another
    assert out["_Lamp"]().hue is out["_Level"]().hue is out["_Lamp"].model_validate({"hue": "red"}).hue
    assert (type(out["_Shade"]().child).__name__, out["_Shade"]().child.x) == ("_Child", 2)  # not a dict


class _Thought(dspy.Type):
    text: str = ""

    def format(self):
        return self.text


def test_adapters_naming_types_of_the_user_keep_them():
    from dspy.primitives.facade import _adapter_spec
    from dspy.primitives.sandbox_guest import rebuild_adapter

    spec, named = _adapter_spec(dspy.ChatAdapter(native_response_types=[_Thought]))
    rebuilt = rebuild_adapter(spec, None, rebuild_types(types_spec(named)))
    assert [t.__name__ for t in rebuilt.native_response_types] == ["_Thought"]
    assert rebuilt.native_response_types[0] is not _Thought  # the sandbox's own copy


class _Prefixing(dspy.ChatAdapter):
    def __init__(self, api_key: str = "", lm=None):
        super().__init__()
        self.api_key = api_key  # state an adapter of the user's might hold
        self.lm = lm

    def format_task_description(self, signature):
        return f"PREFIX {self.api_key[:2]} " + super().format_task_description(signature)


def test_an_adapter_class_of_the_user_runs_on_the_host_with_its_state():
    from dspy.primitives.facade import SandboxSession, _adapter_spec
    from dspy.primitives.sandbox_guest import HostAdapter, rebuild_adapter, signature_spec

    adapter = _Prefixing(api_key="sk-CANARY-1", lm=dspy.LM("openai/m", api_key="sk-CANARY-2"))
    assert _adapter_spec(adapter) == ({"host": True}, {})  # nothing of it crosses
    assert isinstance(rebuild_adapter({"host": True}), HostAdapter)

    lm = DummyLM([{"answer": "pong"}])
    call = {"signature": signature_spec(dspy.Signature("question -> answer")), "inputs": {"question": "ping"}, "demos": []}
    with dspy.context(lm=lm, adapter=adapter):
        session = SandboxSession({})
        session._spec("session")
        assert session._serve_lm("adapter", prompt=call, options={}) == [{"answer": "pong"}]
        with pytest.raises(TypeError, match="may not set LM option"):  # the same metered LM, with the same allowlist
            session._serve_lm("adapter", prompt=call, options={"api_base": "http://elsewhere"})
    assert "PREFIX sk" in str(lm.history[-1]["messages"])  # formatted by the user's adapter, with its state


class _Recording(dspy.ChatAdapter):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        self.calls.append(("sync", demos, lm_kwargs))
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        self.calls.append(("async", demos, lm_kwargs))
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)


def test_a_user_adapter_on_the_host_gets_typed_demos_and_its_async_path():
    import asyncio

    from dspy.primitives.facade import SandboxSession
    from dspy.primitives.sandbox_guest import HostAdapter, HostLM

    class Verdict(pydantic.BaseModel):  # as sandboxed code defines one
        ok: bool
        why: str

    Verdict.__module__ = "__main__"
    signature = dspy.Signature({"case": (_Child, dspy.InputField()), "answer": (str, dspy.OutputField())})
    adapter = _Recording()
    lm = DummyLM([{"answer": "a"}, {"answer": "b"}, {"verdict": '{"ok": true, "why": "fine"}'}], adapter=dspy.ChatAdapter())
    with dspy.context(lm=lm, adapter=adapter):
        session = SandboxSession({}, custom_types={"_Child": _Child})
        session._spec("session")
        host = HostAdapter(lambda **call: session._serve_lm(**call))
        sandbox_lm = HostLM(lambda **call: None, {"model": "m", "model_type": "chat", "kwargs": {},
                            "capabilities": {"supports_function_calling": False, "supports_reasoning": False,
                                             "supports_response_schema": False, "supported_params": []}}).copy(cache=False)
        demos = [dspy.Example(case=_Child(x=1), answer="no")]
        host(sandbox_lm, {}, signature, demos, {"case": _Child(x=2)})
        asyncio.run(host.acall(sandbox_lm, {}, signature, demos, {"case": _Child(x=3)}))
        # A type sandboxed code defined: the host formats it as text, which is parsed into it here.
        out = host(sandbox_lm, {}, dspy.Signature({"verdict": (Verdict, dspy.OutputField())}), [], {})
    (kind, sent_demos, options), (async_kind, _, _), _ = adapter.calls
    assert (kind, async_kind) == ("sync", "async")  # an async predictor runs the adapter's acall
    assert type(sent_demos[0].case).__name__ == "_Child" and sent_demos[0].case.x == 1  # demos typed again
    assert options.get("cache") is False  # the copy's cache choice comes along
    assert isinstance(out[0]["verdict"], Verdict) and out[0]["verdict"].why == "fine"


def test_an_async_adapter_call_keeps_the_callers_settings():
    import asyncio

    from dspy.primitives.facade import _run_coroutine

    async def read():
        return dspy.settings.lm

    async def scoped():
        with dspy.context(lm="scoped"):
            return _run_coroutine(read())  # from a thread that already runs a loop

    assert asyncio.run(scoped()) == "scoped"


def test_values_cross_as_pydantic_dumps_them_by_field_name():
    class Thing:
        def __str__(self):
            return "thing"

    @dataclasses.dataclass
    class Holder:
        thing: Thing

    class Secret(pydantic.BaseModel):
        payload: bytes
        token: str = pydantic.Field(default="", exclude=True)

    assert encode_value({_Hue.RED: {_Spot(1)}, "image": dspy.Image("https://example.com/a.png")}) == {
        "red": [{"x": 1, "label": ""}], "image": {"url": "https://example.com/a.png"},
    }
    assert encode_value(Holder(Thing())) == {"thing": "thing"}
    assert encode_value({("a", "b"): 1, 2: "two"}) == {'["a", "b"]': 1, 2: "two"}  # JSON keys
    with pytest.raises(ValueError, match="distinct"):
        encode_value({("a", "b"): 1, '["a", "b"]': 2})  # rather than dropping one of them

    class Key(pydantic.BaseModel, frozen=True):
        name: str
        token: str = pydantic.Field(default="", exclude=True)

    assert encode_value({Key(name="k", token="secret"): 1}) == {'{"name": "k"}': 1}
    assert encode_value(Secret(payload=b"ok", token="secret")) == {"payload": "ok"}
    with pytest.raises((pydantic_core.PydanticSerializationError, UnicodeDecodeError)):
        encode_value(Secret(payload=b"\xff", token="secret"))  # not sent at all, rather than some other way


def test_two_step_adapter_runs_its_extraction_model_through_the_host():
    import json

    from dspy.primitives.sandbox_guest import rebuild_adapter

    main = DummyLM([{"answer": "the capital is Paris"}], adapter=dspy.ChatAdapter())
    extraction = DummyLM([{"answer": "Paris"}], adapter=dspy.ChatAdapter())
    adapter = dspy.TwoStepAdapter(extraction, use_native_function_calling=False)
    with dspy.context(adapter=adapter):
        session = SandboxSession({}, lm=main)
        spec = session._spec("test")
        rebuilt = rebuild_adapter(json.loads(json.dumps(spec["adapter"])), session._serve_lm)
        host = HostLM(session._serve_lm, spec["lm"])
        assert type(rebuilt) is dspy.TwoStepAdapter and rebuilt.use_native_function_calling is False
        with dspy.context(lm=host, adapter=rebuilt):
            assert dspy.Predict("question -> answer")(question="Capital of France?").answer == "Paris"
    assert len(main.history) == len(extraction.history) == 1  # both stages ran on the host's own LMs


@pytest.mark.parametrize("where", ["message", "tool_result", "system", "prompt"])
def test_sandboxed_requests_cannot_have_the_host_read_its_files(tmp_path, where):
    from dspy._vendor.lm15.types import DocumentPart, TextPart, ToolResultPart

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    lm = _lm()
    session = SandboxSession({}, lm=lm)
    host = _host_lm(session)
    document = DocumentPart(media_type="text/plain", path=secret)
    with pytest.raises(PermissionError, match="host files"):
        if where == "prompt":
            host._build_adapter_request(Prompt(messages=(Message.user((document,)),)), {})
        elif where == "system":
            host(Request(model=lm.model, system=(document,), messages=(Message.user("hi"),)))
        elif where == "tool_result":
            result = ToolResultPart(id="call-1", content=(TextPart(text="see"), document))
            host(Request(model=lm.model, messages=(Message.user("hi"), Message(role="tool", parts=(result,)))))
        else:
            host(Request(model=lm.model, messages=(Message.user((TextPart(text="read this"), document)),)))
    assert lm.history == []
