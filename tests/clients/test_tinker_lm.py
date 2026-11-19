"""TinkerLM against a fake Tinker SDK and renderer: no tinker install, network or key."""

import importlib.util
import json
import pickle
import re
import sys
import types
from dataclasses import dataclass, field

import pytest

import dspy
from dspy.clients.tinker import AsyncTinkerEngine, TinkerEngine
from dspy.lm15 import (
    BuiltinTool,
    Config,
    FunctionTool,
    ImagePart,
    Message,
    Request,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
)
from tests.clients.test_engine_errors import invoke

STOP_TOKEN = 7


@dataclass
class SamplingParams:
    max_tokens: int | None = None
    seed: int | None = None
    stop: list | None = None
    temperature: float = 1
    top_k: int = -1
    top_p: float = 1


@dataclass
class FunctionBody:
    name: str
    arguments: str


@dataclass
class ToolCall:
    function: FunctionBody
    id: str | None = None


ToolCall.FunctionBody = FunctionBody


@dataclass
class Prompt:
    messages: list
    length: int


class Future:
    def __init__(self, value):
        self.value = value
        self.awaited = False

    def result(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    def __await__(self):
        self.awaited = True
        return self._get().__await__()

    async def _get(self):
        return self.result()


class Renderer:
    """Renders messages as they are, and parses Qwen-style think and tool_call tags."""

    def __init__(self, tools=True):
        self.tools = tools
        self.prompts = []

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        if not self.tools:
            raise NotImplementedError
        return [{"role": "system", "content": system_prompt + "\n\n# Tools\n" + json.dumps(tools)}]

    def get_stop_sequences(self):
        return [STOP_TOKEN]

    def build_generation_prompt(self, messages):
        self.prompts.append(messages)
        return Prompt(messages, length=10 * len(messages))

    def parse_response(self, tokens):
        text = "".join(map(chr, tokens))
        if text.count("<think>") != text.count("</think>"):
            return {"role": "assistant", "content": text}, "malformed"
        if "<think>" not in text and "<tool_call>" not in text:
            return {"role": "assistant", "content": text}, "stop_sequence"
        parts, calls, unparsed = [], [], []
        pattern = r"<think>(.*?)</think>|<tool_call>(.*?)</tool_call>|((?:(?!<think>|<tool_call>).)+)"
        for match in re.finditer(pattern, text, re.S):
            think, call, plain = match.groups()
            if think is not None:
                parts.append({"type": "thinking", "thinking": think})
            elif plain is not None:
                parts.append({"type": "text", "text": plain})
            else:
                try:
                    body = json.loads(call)
                    function = FunctionBody(name=body["name"], arguments=json.dumps(body["arguments"]))
                    calls.append(ToolCall(function=function, id=body.get("id")))
                except (ValueError, KeyError):
                    unparsed.append(types.SimpleNamespace(raw_text=match.group(0)))
        message = {"role": "assistant", "content": parts}
        if calls:
            message["tool_calls"] = calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, "stop_sequence"


@dataclass
class SamplingClient:
    base_model: str = "Qwen/Qwen3-8B"
    replies: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    futures: list = field(default_factory=list)

    def reply(self, text, stop_reason="stop"):
        self.replies.append((text, stop_reason))

    def sample(self, prompt, num_samples, sampling_params):
        self.calls.append({"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            future = Future(reply)
        else:
            text, stop_reason = reply
            sequence = types.SimpleNamespace(tokens=[ord(c) for c in text], stop_reason=stop_reason, sequence_id="s-1")
            future = Future(types.SimpleNamespace(sequences=[sequence]))
        self.futures.append(future)
        return future

    def get_base_model(self):
        return self.base_model

    def get_tokenizer(self):
        return "tokenizer"


def _fake_errors(module):
    class TinkerError(Exception):
        pass

    class APIError(TinkerError):
        def __init__(self, message):
            super().__init__(message)
            self.message = message

    class APIStatusError(APIError):
        status_code = 500

    class APIConnectionError(APIError):
        pass

    class RequestFailedError(TinkerError):
        def __init__(self, message, request_id, category):
            super().__init__(message)
            self.message, self.request_id, self.category = message, request_id, category

    module.TinkerError, module.APIError, module.APIStatusError = TinkerError, APIError, APIStatusError
    module.APIConnectionError, module.RequestFailedError = APIConnectionError, RequestFailedError
    module.APITimeoutError = type("APITimeoutError", (APIConnectionError,), {})
    for name, status in [
        ("AuthenticationError", 401),
        ("PermissionDeniedError", 403),
        ("BillingError", 402),
        ("NotFoundError", 404),
        ("BadRequestError", 400),
        ("UnprocessableEntityError", 422),
        ("ConflictError", 409),
        ("RateLimitError", 429),
        ("InternalServerError", 500),
    ]:
        setattr(module, name, type(name, (APIStatusError,), {"status_code": status}))


@pytest.fixture
def sdk(monkeypatch):
    """Install a fake `tinker` SDK and `tinker_cookbook` and return their recorders."""
    tinker = types.ModuleType("tinker")
    tinker.SamplingParams = SamplingParams
    _fake_errors(tinker)
    state = types.SimpleNamespace(
        client=SamplingClient(), renderer=Renderer(), services=[], renderers=[], tinker=tinker
    )

    class ServiceClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            state.services.append(self)

        def create_sampling_client(self, model_path=None, base_model=None):
            self.target = (model_path, base_model)
            return Future(state.client)

    tinker.ServiceClient = ServiceClient
    cookbook = types.ModuleType("tinker_cookbook")
    cookbook.__version__ = "0.7.0"
    renderers = types.ModuleType("tinker_cookbook.renderers")
    renderers.ToolCall = ToolCall

    def get_renderer(name, tokenizer, model_name=None):
        state.renderers.append((name, tokenizer, model_name))
        return state.renderer

    renderers.get_renderer = get_renderer
    model_info = types.ModuleType("tinker_cookbook.model_info")
    model_info.get_recommended_renderer_name = lambda model: {"Qwen/Qwen3-8B": "qwen3"}[model]
    for name, module in [
        ("tinker", tinker),
        ("tinker_cookbook", cookbook),
        ("tinker_cookbook.renderers", renderers),
        ("tinker_cookbook.model_info", model_info),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return state


def request(*messages, system=None, tools=(), **config):
    return Request(
        model="tinker/Qwen/Qwen3-8B",
        messages=messages or (Message.user("hi"),),
        system=system,
        tools=tools,
        config=Config(**config),
    )


SEARCH = FunctionTool(
    name="search", description="Search the web.", parameters={"type": "object", "properties": {"q": {"type": "string"}}}
)


def test_conversation_renders_and_sampling_params_pass_through(sdk):
    sdk.client.reply("done")
    conversation = (
        Message.user("Find X."),
        Message.assistant(
            [ThinkingPart("look it up"), TextPart("Searching."), ToolCallPart(id="c1", name="search", input={"q": "X"})]
        ),
        Message(role="tool", parts=(ToolResultPart(id="c1", content=(TextPart("X is 4"),)),)),
        Message.assistant("X is 4."),
        Message.user("Thanks."),
    )
    response = TinkerEngine("Qwen/Qwen3-8B", sampling_client=sdk.client).complete(
        request(
            *conversation,
            system="Be brief.",
            temperature=0.3,
            max_tokens=64,
            top_p=0.9,
            top_k=20,
            seed=5,
        )
    )

    assert response.text == "done"
    assert sdk.renderers == [("qwen3", "tokenizer", "Qwen/Qwen3-8B")]
    assert sdk.renderer.prompts == [
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Find X."},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "look it up"}, {"type": "text", "text": "Searching."}],
                "tool_calls": [ToolCall(FunctionBody("search", '{"q": "X"}'), id="c1")],
            },
            {"role": "tool", "content": "X is 4", "tool_call_id": "c1", "name": "search"},
            {"role": "assistant", "content": "X is 4."},
            {"role": "user", "content": "Thanks."},
        ]
    ]
    [call] = sdk.client.calls
    assert call["num_samples"] == 1
    assert call["sampling_params"] == SamplingParams(
        max_tokens=64, seed=5, stop=[STOP_TOKEN], temperature=0.3, top_k=20, top_p=0.9
    )


def test_tools_are_rendered_into_the_system_prefix(sdk):
    sdk.client.reply("ok")
    TinkerEngine("Qwen/Qwen3-8B", sampling_client=sdk.client).complete(request(system="Be brief.", tools=(SEARCH,)))
    system, user = sdk.renderer.prompts[0]
    spec = {"name": "search", "description": "Search the web.", "parameters": SEARCH.parameters}
    assert system == {"role": "system", "content": "Be brief.\n\n# Tools\n" + json.dumps([spec])}
    assert user == {"role": "user", "content": "hi"}


@pytest.mark.parametrize(
    ("sampled", "stop_reason", "parts", "finish_reason"),
    [
        ("<think>capital?</think>Paris", "stop", [ThinkingPart("capital?"), TextPart("Paris")], "stop"),
        (
            '<tool_call>{"name": "search", "arguments": {"q": "X"}}</tool_call>',
            "stop",
            [ToolCallPart(id="call_0", name="search", input={"q": "X"})],
            "tool_call",
        ),
        (
            '<tool_call>{"name": "search", "arguments": {"q": "X"}, "id": "t9"}</tool_call>',
            "stop",
            [ToolCallPart(id="t9", name="search", input={"q": "X"})],
            "tool_call",
        ),
        ("<think>still thinking", "length", [TextPart("<think>still thinking")], "length"),
        ("<tool_call>{broken</tool_call>", "stop", [TextPart("<tool_call>{broken</tool_call>")], "stop"),
    ],
)
def test_sampled_tokens_parse_into_parts_finish_reason_and_usage(sdk, sampled, stop_reason, parts, finish_reason):
    sdk.client.reply(sampled, stop_reason)
    response = TinkerEngine("Qwen/Qwen3-8B", sampling_client=sdk.client).complete(request())
    assert list(response.message.parts) == parts
    assert response.finish_reason == finish_reason
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == len(sampled)


def test_request_stop_trims_the_reply_and_is_recorded(sdk):
    sdk.client.reply("one END two")
    response = TinkerEngine("Qwen/Qwen3-8B", sampling_client=sdk.client).complete(request(stop=("END",)))
    assert response.text == "one "
    assert response.usage.output_tokens == len("one END two")
    assert sdk.client.calls[0]["sampling_params"].stop == [STOP_TOKEN]
    [note] = response.adaptations
    assert (note.field, note.action, note.asked) == ("config.stop", "client_side", ["END"])


def test_named_tool_choice_renders_only_that_tool_and_records_it(sdk):
    sdk.client.reply("ok")
    other = FunctionTool(name="submit")
    choice = ToolChoice(mode="required", allowed=("submit",))
    response = TinkerEngine("Qwen/Qwen3-8B", sampling_client=sdk.client).complete(
        request(tools=(SEARCH, other), tool_choice=choice),
    )
    system = sdk.renderer.prompts[0][0]["content"]
    assert '"submit"' in system and '"search"' not in system
    [note] = response.adaptations
    assert (note.field, note.asked, note.applied) == ("config.tool_choice", "required", "auto")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_lm_samples_through_the_matching_sdk_path(sdk, asynchronous):
    sdk.client.reply("hello")
    lm = dspy.TinkerLM(sampling_client=sdk.client)
    assert await invoke(lm, asynchronous, "hi", temperature=0.2) == ["hello"]
    assert lm.model == "tinker/Qwen/Qwen3-8B"
    assert lm.cache is False
    assert sdk.client.futures[0].awaited is asynchronous
    assert sdk.client.calls[0]["sampling_params"].temperature == 0.2
    assert lm.history[0]["usage"]["prompt_tokens"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sampling_client_is_created_once_from_the_checkpoint(sdk, asynchronous, monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    sdk.client.reply("a")
    sdk.client.reply("b")
    path = "tinker://run-1/sampler_weights/final"
    lm = dspy.TinkerLM(model_path=path, api_key="tml-secret", cache=False)
    assert not sdk.services
    assert await invoke(lm, asynchronous, "one") == ["a"]
    assert await invoke(lm, not asynchronous, "two") == ["b"]
    [service] = sdk.services
    assert service.api_key == "tml-secret"
    assert service.target == (path, None)
    assert sdk.renderers == [("qwen3", "tokenizer", "Qwen/Qwen3-8B")]
    assert lm.model == f"tinker/{path}"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_predict_with_chat_adapter(sdk, asynchronous):
    sdk.client.reply("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]")
    program = dspy.Predict("question -> answer")
    with dspy.context(
        lm=dspy.TinkerLM("Qwen/Qwen3-8B", cache=False, sampling_client=sdk.client),
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
    ):
        result = (
            await program.acall(question="Capital of France?")
            if asynchronous
            else program(question="Capital of France?")
        )
    assert result.answer == "Paris"
    system, user = sdk.renderer.prompts[0]
    assert system["role"] == "system" and "`answer`" in system["content"]
    assert "Capital of France?" in user["content"]


def test_predict_with_native_function_calling(sdk):
    def search(q: str) -> str:
        """Search the web."""
        return q

    class Plan(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        calls: dspy.ToolCalls = dspy.OutputField()

    sdk.client.reply(
        "[[ ## answer ## ]]\nlooking\n\n[[ ## completed ## ]]"
        '<tool_call>{"name": "search", "arguments": {"q": "X"}}</tool_call>'
    )
    lm = dspy.TinkerLM("Qwen/Qwen3-8B", cache=False, sampling_client=sdk.client)
    with dspy.context(
        lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True, use_json_adapter_fallback=False)
    ):
        result = dspy.Predict(Plan)(question="What is X?", tools=[dspy.Tool(search)])
    assert [(call.name, call.args) for call in result.calls.tool_calls] == [("search", {"q": "X"})]
    assert result.answer == "looking"
    assert '"name": "search"' in sdk.renderer.prompts[0][0]["content"]


def test_react_forced_submit_offers_only_submit(sdk):
    def population(city: str) -> str:
        """Return the population of a city."""
        return "2.1 million"

    sdk.client.reply('<tool_call>{"name": "population", "arguments": {"city": "Paris"}}</tool_call>')
    sdk.client.reply('<tool_call>{"name": "submit", "arguments": {"answer": "2.1 million"}}</tool_call>')
    lm = dspy.TinkerLM("Qwen/Qwen3-8B", cache=False, sampling_client=sdk.client)
    with dspy.context(lm=lm):
        result = dspy.ReAct("question -> answer", tools=[population], max_iters=1)(question="Population of Paris?")
    assert result.answer == "2.1 million"
    first, forced = (prompt[0]["content"] for prompt in sdk.renderer.prompts)
    assert '"name": "population"' in first
    assert '"name": "submit"' in forced and '"name": "population"' not in forced
    assert [note.field for note in lm.history[-1]["adaptations"]] == ["config.tool_choice"]


def test_state_round_trip_keeps_config_and_no_secret(sdk):
    lm = dspy.TinkerLM(
        "Qwen/Qwen3-8B",
        model_path="tinker://run-1/sampler_weights/final",
        renderer="qwen3_disable_thinking",
        api_key="tml-secret",
        temperature=0.3,
        num_retries=1,
    )
    state = lm.dump_state()
    assert "tml-secret" not in json.dumps(state)
    assert {key: state[key] for key in ("model", "model_path", "renderer", "temperature", "num_retries")} == {
        "model": "Qwen/Qwen3-8B",
        "model_path": "tinker://run-1/sampler_weights/final",
        "renderer": "qwen3_disable_thinking",
        "temperature": 0.3,
        "num_retries": 1,
    }
    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.BaseLM.load_state(state)
    restored = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)
    assert type(restored) is dspy.TinkerLM
    assert restored.model == "tinker/tinker://run-1/sampler_weights/final?renderer=qwen3_disable_thinking"
    assert restored.dump_state() == state


def test_copied_programs_share_the_engine_and_pickles_drop_the_key(sdk, caplog):
    sdk.client.reply("ok")
    program = dspy.Predict("question -> answer")
    program.set_lm(dspy.TinkerLM("Qwen/Qwen3-8B", api_key="tml-secret", cache=False))
    program.lm("hi")
    assert program.deepcopy().lm.engine is program.lm.engine
    assert "Failed to deep copy" not in caplog.text
    restored = pickle.loads(pickle.dumps(program.lm)).engine
    assert restored.dump_state() == program.lm.engine.dump_state()
    # The SDK pickles a sampling client with its API key, so the created one is dropped.
    assert (restored._api_key, restored._client, restored._renderer) == (None, None, None)


def test_engine_pair_saves_through_dspy_lm(sdk):
    engine = TinkerEngine("Qwen/Qwen3-8B", renderer="qwen3")
    lm = dspy.LM("tinker/Qwen/Qwen3-8B", engine=engine, async_engine=AsyncTinkerEngine(engine))
    restored = dspy.LM.load_state(lm.dump_state(), allow_custom_lm_class=True)
    assert isinstance(restored.engine, TinkerEngine) and isinstance(restored.async_engine, AsyncTinkerEngine)
    assert restored.async_engine.dump_state() == {"model": "Qwen/Qwen3-8B", "model_path": None, "renderer": "qwen3"}


def test_missing_sdk_names_the_extra(monkeypatch):
    find_spec = importlib.util.find_spec
    for name in ("tinker", "tinker_cookbook"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *args: None if name.startswith("tinker") else find_spec(name, *args)
    )
    lm = dspy.TinkerLM("Qwen/Qwen3-8B", cache=False)
    with pytest.raises(ImportError, match=r"tinker is required to use dspy\.TinkerLM.*\[tinker\]"):
        lm("hi")


@pytest.mark.parametrize(
    ("prompt", "tool_format", "feature"),
    [
        (request(Message.user([TextPart("see"), ImagePart(url="https://example.com/a.png")])), True, "image"),
        (request(Message.developer("be terse")), True, "developer"),
        (request(response_format={"type": "json_object"}), True, "response_format"),
        (request(logprobs=0), True, "logprobs"),
        (request(tools=(BuiltinTool(name="web_search"),)), True, "tools"),
        (request(tools=(SEARCH,)), False, "tools"),
    ],
)
def test_unsupported_inputs_are_refused_before_sampling(sdk, prompt, tool_format, feature):
    sdk.renderer.tools = tool_format
    lm = dspy.TinkerLM(sampling_client=sdk.client, cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError) as info:
        lm(prompt)
    assert info.value.feature == feature
    assert "Tinker" in str(info.value)
    assert not sdk.client.calls


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (lambda t: t.AuthenticationError("bad key"), dspy.LMAuthError),
        (lambda t: t.RateLimitError("slow down"), dspy.LMRateLimitError),
        (lambda t: t.NotFoundError("no such checkpoint"), dspy.LMUnsupportedModelError),
        (lambda t: t.BadRequestError("bad prompt"), dspy.LMInvalidRequestError),
        (lambda t: t.InternalServerError("boom"), dspy.LMServerError),
        (lambda t: t.APITimeoutError("late"), dspy.LMTimeoutError),
        (lambda t: t.APIConnectionError("down"), dspy.LMTransportError),
        (lambda t: t.RequestFailedError("rejected", "r-1", "user"), dspy.LMInvalidRequestError),
        (lambda t: RuntimeError("engine bug"), dspy.LMUnexpectedError),
    ],
)
def test_sdk_failures_map_to_dspy_errors(sdk, error, expected):
    cause = error(sdk.tinker)
    sdk.client.replies.append(cause)
    lm = dspy.TinkerLM(sampling_client=sdk.client, cache=False, num_retries=0)
    with pytest.raises(expected) as info:
        lm("hi")
    # Expected failures pass through one dspy.lm15 error; an unexpected one is wrapped directly.
    chain = info.value.__cause__ if expected is dspy.LMUnexpectedError else info.value.__cause__.__cause__
    assert chain is cause
