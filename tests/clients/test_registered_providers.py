"""dspy.lm15.register_provider: an HTTP provider lm15's registry does not
list becomes a native route for every dspy.LM constructed afterwards, with
the LM's client settings honored — the alternative to writing a custom
engine for it (cmpnd-ai/breaka-your-lm gauntlet/fireworks.py, 2026-09-16).

The rules pinned here came out of review of the first design (dspy#10441):
one binding per LM, a fallback that keeps the declared destination and
credential, and aliases that grant nothing but a spelling.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import dspy
from dspy.clients.backend_selection import select_backend
from dspy.clients.execution import _select_engine, prepare
from dspy.lm15 import (
    AccessPolicy,
    EndpointSupport,
    ModelSupport,
    OpenAIChatCompat,
    ProviderDefinition,
    RegisteredProvider,
    register_provider,
    registered_providers,
    unregister_provider,
)

MODEL = "accounts/fireworks/models/deepseek-v4-flash"  # in the bundled metadata snapshot (tools, reasoning, schema, prices)


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"auth": self.headers.get("Authorization"), "path": self.path, "body": body,
                         "headers": dict(self.headers)})
            out = {"id": "x", "object": "chat.completion", "model": body["model"],
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1", seen
    finally:
        http.shutdown()
        http.server_close()
        worker.join()


def _definition(base_url, provider="fireworks", aliases=("fireworks-ai",), headers=(), auth_scheme=("bearer",), **compat):
    compat.setdefault("max_tokens_field", "max_tokens")
    return ProviderDefinition.chat(
        AccessPolicy(provider=provider, supports=EndpointSupport(complete=True, stream=True, models=True),
                     auth_modes=("bearer",), auth_scheme=auth_scheme, env_keys=("FIREWORKS_API_KEY",),
                     base_url=base_url, headers=tuple(headers)),
        compat=OpenAIChatCompat(**compat),
        aliases=aliases, note="Fireworks (test)",
    )


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    import dspy.clients.model_metadata as metadata

    for name in ("FIREWORKS_API_KEY", "FIREWORKS_API_BASE", "FIREWORKS_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Capabilities and prices come from the bundled metadata snapshot, not
    # from whatever the live LiteLLM map says today (or whichever a sibling
    # test loaded first).
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(metadata, "_data", None)
    monkeypatch.setattr(metadata, "_source", {})
    before = {b.id for b in registered_providers()}
    yield
    for binding in registered_providers():
        if binding.id not in before:
            unregister_provider(binding.id)


@pytest.fixture
def litellm_stub(monkeypatch):
    """LiteLLM as the fallback would call it: records the arguments, answers."""
    import litellm

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return litellm.ModelResponse(**{
            "id": "x", "object": "chat.completion", "model": kwargs["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async def acompletion(**kwargs):
        return completion(**kwargs)

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


# ─── the registry ─────────────────────────────────────────────────────


def test_register_is_idempotent_and_replace_is_explicit():
    definition = _definition("https://api.fireworks.test/v1")
    binding = register_provider(definition)
    assert isinstance(binding, RegisteredProvider) and binding.definition is definition
    assert register_provider(definition) is binding  # same registration: no-op
    with pytest.raises(ValueError, match="already registered with a different registration"):
        register_provider(_definition("https://other.test/v1"))
    with pytest.raises(ValueError, match="already registered with a different registration"):
        register_provider(definition, metadata_namespaces=("fireworks_ai",))  # statements are part of it
    replaced = register_provider(_definition("https://other.test/v1"), replace=True)
    assert registered_providers() == (replaced,)
    unregister_provider("fireworks")
    assert registered_providers() == ()
    unregister_provider("fireworks")  # unknown: ignored


def test_arguments_are_type_checked():
    definition = _definition("https://x.test/v1")
    with pytest.raises(TypeError, match="ProviderDefinition"):
        register_provider("fireworks")
    with pytest.raises(TypeError, match="ModelSupport"):
        register_provider(definition, supports={"function_calling": True})
    with pytest.raises(TypeError, match="models= maps"):
        register_provider(definition, models={MODEL: True})
    with pytest.raises(TypeError, match="metadata_namespaces"):
        register_provider(definition, metadata_namespaces="fireworks_ai")
    with pytest.raises(TypeError, match="True, False or None"):
        ModelSupport(function_calling="yes")


def test_spellings_lm15_or_litellm_already_use_are_refused():
    for taken in ("groq", "ollama-chat"):
        with pytest.raises(Exception, match="already names"):
            register_provider(_definition("https://x.test/v1", provider="new-door", aliases=(taken,)))
    register_provider(_definition("https://x.test/v1"))
    with pytest.raises(ValueError, match=r"spells \['fireworks-ai'\] like the registered provider 'fireworks'"):
        register_provider(_definition("https://x.test/v1", provider="other", aliases=("fireworks-ai",)))


# ─── one binding per LM ───────────────────────────────────────────────


def test_an_lm_binds_the_registrations_present_at_construction(server):
    base, _ = server
    before = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False)
    register_provider(_definition(base))
    after = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    assert not select_backend(before).native  # bound nothing; stays as it was
    assert select_backend(after).native
    assert after("hi") == ["Paris"]
    assert after.copy(temperature=0.5)._providers is after._providers  # copies share the binding
    after.close()


def test_replacing_a_registration_never_splits_an_lm(server):
    base, _ = server
    register_provider(_definition(base, max_tokens_field="max_tokens"))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    call = prepare(lm, "hi", None, {"max_tokens": 7})
    sync_engine, _, _ = _select_engine(lm, call, False)
    register_provider(_definition("http://127.0.0.1:9/v1", max_tokens_field="max_completion_tokens"), replace=True)
    # Selection, the cached sync engine and a fresh async engine all read the
    # LM's own binding, not the registry as it is now.
    assert select_backend(lm).resolution.compat.max_tokens_field == "max_tokens"
    same, _, _ = _select_engine(lm, call, False)
    assert same is sync_engine and same.config.providers[0].access.base_url == base

    async def async_engine():
        engine, _, _ = _select_engine(lm, call, True)
        return engine.config.providers[0].access.base_url

    assert asyncio.run(async_engine()) == base
    assert lm("hi") == ["Paris"]  # still the address it was built with
    fresh = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    assert select_backend(fresh).resolution.compat.max_tokens_field == "max_completion_tokens"
    lm.close()


# ─── routing through dspy.LM ──────────────────────────────────────────


@pytest.mark.parametrize("prefix", ["fireworks", "fireworks_ai", "fireworks-ai"])
def test_registered_provider_routes_natively_with_the_lm_key(server, prefix):
    base, seen = server
    register_provider(_definition(base))
    lm = dspy.LM(f"{prefix}/{MODEL}", api_key="lm-key", cache=False, num_retries=0)
    selection = select_backend(lm)
    assert selection.native and selection.resolution.provider == "fireworks" and selection.resolution.declared
    assert lm("capital of France?") == ["Paris"]
    assert seen[-1]["auth"] == "Bearer lm-key"
    assert seen[-1]["path"] == "/v1/chat/completions"
    assert seen[-1]["body"]["model"] == MODEL
    lm.close()


def test_env_key_and_client_settings_are_honored(server, monkeypatch):
    base, seen = server
    register_provider(_definition(base))
    monkeypatch.setenv("FIREWORKS_API_KEY", "env-key")
    lm = dspy.LM(f"fireworks/{MODEL}", cache=False, num_retries=0, max_tokens=9, timeout=5)
    assert select_backend(lm).native
    assert lm("hi") == ["Paris"]
    assert seen[-1]["auth"] == "Bearer env-key"
    assert seen[-1]["body"]["max_tokens"] == 9  # the declared compat's spelling
    lm.close()
    moved = dspy.LM(f"fireworks/{MODEL}", api_base="http://127.0.0.1:9/v1", cache=False, num_retries=0)
    with pytest.raises(dspy.LMError):
        moved("hi")  # api_base moves the call; a closed port fails, never a silent success elsewhere
    monkeypatch.delenv("FIREWORKS_API_KEY")
    with pytest.raises(dspy.LMError, match="FIREWORKS_API_KEY"):
        dspy.LM(f"fireworks/{MODEL}", cache=False, num_retries=0)("hi")


def test_a_declaration_beats_the_ambient_gateway_variable(server, monkeypatch):
    # {PREFIX}_API_BASE redirects engine="auto" to LiteLLM for registry
    # providers (legacy gateways). A declaration states its address itself.
    base, seen = server
    register_provider(_definition(base))
    monkeypatch.setenv("FIREWORKS_API_BASE", "http://127.0.0.1:9/v1")
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    assert select_backend(lm).native
    assert lm("hi") == ["Paris"] and seen[-1]["path"] == "/v1/chat/completions"
    lm.close()


@pytest.mark.asyncio
async def test_async_path(server):
    base, seen = server
    register_provider(_definition(base))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="lm-key", cache=False, num_retries=0)
    assert await lm.acall("hi") == ["Paris"]
    assert seen[-1]["auth"] == "Bearer lm-key"
    await lm.aclose()


def test_unregistered_prefix_is_not_native():
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False)
    assert not select_backend(lm).native


# ─── the fallback keeps the declared connection ───────────────────────


@pytest.mark.parametrize("prefix", ["fireworks", "fireworks_ai"])
def test_fallback_reaches_the_declared_address_with_the_declared_credential(litellm_stub, prefix, monkeypatch):
    # extra_headers is a client setting only LiteLLM carries. The route it
    # takes is LiteLLM's generic OpenAI-compatible door at the DECLARED
    # address, not LiteLLM's own idea of "fireworks_ai".
    register_provider(_definition("https://private-gateway.test/v1", headers=(("X-Gateway", "tenant-1"),)))
    lm = dspy.LM(f"{prefix}/{MODEL}", api_key="private-key", extra_headers={"X-Test": "1"}, cache=False, num_retries=0)
    selection = select_backend(lm)
    assert not selection.native and selection.resolution.declared
    assert lm("hi")[0].endswith("Paris\n\n[[ ## completed ## ]]")
    sent = litellm_stub[-1]
    assert sent["model"] == f"openai/{MODEL}"
    assert sent["api_base"] == "https://private-gateway.test/v1"
    assert sent["api_key"] == "private-key"
    assert sent["extra_headers"] == {"X-Gateway": "tenant-1", "X-Test": "1"}
    assert lm.history[-1]["model"] == f"{prefix}/{MODEL}"  # the LM's string, not the wire one


def test_fallback_reads_the_declared_key_variable_and_never_an_ambient_openai_key(litellm_stub, monkeypatch):
    register_provider(_definition("https://private-gateway.test/v1"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    lm = dspy.LM(f"fireworks/{MODEL}", extra_headers={"X-Test": "1"}, cache=False, num_retries=0)
    with pytest.raises(dspy.LMError, match="FIREWORKS_API_KEY") as info:
        lm("hi")
    assert not litellm_stub and "sk-openai" not in str(info.value)
    monkeypatch.setenv("FIREWORKS_API_KEY", "env-key")
    assert "Paris" in lm("hi")[0]
    assert litellm_stub[-1]["api_key"] == "env-key"


def test_fallback_under_engine_litellm_takes_the_same_route(litellm_stub):
    register_provider(_definition("https://private-gateway.test/v1"))
    lm = dspy.LM(f"fireworks_ai/{MODEL}", engine="litellm", api_key="k", cache=False, num_retries=0)
    assert "Paris" in lm("hi")[0]
    assert litellm_stub[-1]["model"] == f"openai/{MODEL}"
    assert litellm_stub[-1]["api_base"] == "https://private-gateway.test/v1"


@pytest.mark.asyncio
async def test_async_fallback_keeps_the_connection_too(litellm_stub):
    register_provider(_definition("https://private-gateway.test/v1"))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", extra_headers={"X": "1"}, cache=False, num_retries=0)
    assert "Paris" in (await lm.acall("hi"))[0]
    assert litellm_stub[-1]["api_base"] == "https://private-gateway.test/v1"


def test_capabilities_on_the_fallback_route_come_from_the_declaration(litellm_stub):
    # An adapter asks the LM what it supports before every call; on the
    # fallback route that must not be LiteLLM's reading of a prefix it
    # does not know. The whole program runs.
    register_provider(_definition("https://private-gateway.test/v1"),
                      supports=ModelSupport(function_calling=True, response_schema=True))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", extra_headers={"X": "1"}, cache=False, num_retries=0)
    assert not select_backend(lm).native
    assert lm.supports_function_calling and lm.supports_response_schema and "tools" in lm.supported_params
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        prediction = dspy.Predict("question -> answer")(question="capital of France?")
    assert litellm_stub[-1]["api_base"] == "https://private-gateway.test/v1"
    assert prediction.answer == "Paris"


def test_fallback_sends_the_credential_only_under_the_declared_scheme(litellm_stub):
    # LiteLLM's openai/ door renders the key as a bearer header; its
    # anthropic/ door as x-api-key. A declaration whose scheme the door
    # cannot send gets no fallback — never its secret in an undeclared
    # header (greptile on dspy#10442).
    from dspy.lm15 import AnthropicCompat

    register_provider(_definition("https://private-gateway.test/v1", provider="hdr", aliases=(), auth_scheme=("x-api-key",)))
    lm = dspy.LM(f"hdr/{MODEL}", api_key="secret", extra_headers={"X": "1"}, cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError, match=r"authenticates with 'x-api-key'.*openai/ door cannot send"):
        lm("hi")
    assert not litellm_stub
    register_provider(ProviderDefinition.anthropic(
        AccessPolicy(provider="claude-gw", auth_modes=("x-api-key",), auth_scheme=("x-api-key",),
                     env_keys=("CLAUDE_GW_KEY",), base_url="https://claude-gateway.test"),
        compat=AnthropicCompat()))
    lm = dspy.LM("claude-gw/claude-x", api_key="secret", extra_headers={"X": "1"}, cache=False, num_retries=0)
    assert "Paris" in lm("hi")[0]
    assert litellm_stub[-1]["model"] == "anthropic/claude-x"
    assert litellm_stub[-1]["api_base"] == "https://claude-gateway.test"
    assert litellm_stub[-1]["api_key"] == "secret"
    register_provider(ProviderDefinition.anthropic(
        AccessPolicy(provider="claude-bearer", auth_modes=("bearer",), auth_scheme=("bearer",),
                     env_keys=("CLAUDE_GW_KEY",), base_url="https://claude-gateway.test"),
        compat=AnthropicCompat()))
    lm = dspy.LM("claude-bearer/claude-x", api_key="secret", extra_headers={"X": "1"}, cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError, match=r"authenticates with 'bearer'.*anthropic/ door cannot send"):
        lm("hi")


def test_fallback_resolves_a_callable_credential_per_call(litellm_stub):
    # lm15 credentials may be zero-argument callables (rotating tokens). The
    # fallback invokes them like the native path, on each call, and hands
    # LiteLLM the string (greptile on dspy#10442).
    register_provider(_definition("https://private-gateway.test/v1"))
    tokens = iter(["token-1", "token-2"])
    lm = dspy.LM(f"fireworks/{MODEL}", api_key=lambda: next(tokens), extra_headers={"X": "1"}, cache=False, num_retries=0)
    lm("hi")
    lm("hi again")
    assert [call["api_key"] for call in litellm_stub[-2:]] == ["token-1", "token-2"]


@pytest.mark.asyncio
async def test_fallback_is_still_the_declared_provider_for_pricing(litellm_stub):
    # The fallback reaches the same server; pricing comes from the
    # declaration's namespaces on every path, never from LiteLLM's reading
    # of the generic door's model string (greptile on dspy#10442).
    register_provider(_definition("https://private-gateway.test/v1"), metadata_namespaces=("fireworks_ai",))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", extra_headers={"X": "1"}, cache=False, num_retries=0)
    lm("hi")  # legacy body path
    lm(dspy.lm15.Request(model=lm.model, messages=(dspy.lm15.Message.user("hi"),)))  # canonical path
    await lm.acall("hi")  # async legacy path
    costs = [entry["cost"] for entry in lm.history[-3:]]
    assert all(cost is not None and cost > 0 for cost in costs) and len(set(costs)) == 1
    for entry in lm.history[-3:]:
        assert entry["cost_details"]["provider"] == "fireworks"
        assert entry["cost_details"]["metadata"]["namespaces"] == ["fireworks_ai"]
    # A colliding name on a gateway with no namespace is unknown, not OpenAI's price.
    register_provider(_definition("https://private-gateway.test/v1", provider="gw", aliases=()))
    plain = dspy.LM("gw/gpt-4o", api_key="k", extra_headers={"X": "1"}, cache=False, num_retries=0)
    plain("hi")
    assert plain.history[-1]["cost"] is None
    assert "names no metadata namespace" in plain.history[-1]["cost_details"]["reason"]


def test_a_declared_refusal_is_final_under_auto(litellm_stub, server):
    # thinking_format="none" says the wire has no reasoning field. A
    # reasoning dial on it is a refusal (lm15 MAP-7); sending it through
    # LiteLLM anyway would drop the dial silently, so engine="auto" refuses
    # too. (A setting lm15 adapts and records, e.g. json_schema="reject",
    # is not a refusal and never leaves the native route.)
    base, seen = server
    register_provider(_definition(base, thinking_format="none", json_schema="reject"))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError):
        lm("hi", reasoning_effort="low")
    assert not litellm_stub
    assert lm("hi", response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}) == ["Paris"]
    assert "response_format" not in seen[-1]["body"]  # adapted and recorded, natively
    lm.close()


def test_engine_lm15_refuses_client_settings_it_cannot_carry(server):
    base, _ = server
    register_provider(_definition(base))
    with pytest.raises(Exception, match="require the LiteLLM"):
        select_backend(dspy.LM(f"fireworks/{MODEL}", engine="lm15", api_key="k", extra_headers={"X": "1"}))


# ─── aliases grant a spelling, nothing else ───────────────────────────


def test_an_alias_grants_no_metadata(server):
    base, _ = server
    register_provider(_definition(base))  # alias fireworks-ai, no namespaces
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    assert not (lm.supports_function_calling or lm.supports_reasoning or lm.supports_response_schema)
    lm("hi")
    assert lm.history[-1]["cost"] is None
    assert "names no metadata namespace" in lm.history[-1]["cost_details"]["reason"]
    lm.close()


def test_metadata_namespaces_are_an_explicit_opt_in(server):
    base, _ = server
    register_provider(_definition(base), metadata_namespaces=("fireworks_ai",))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False, num_retries=0)
    assert lm.supports_function_calling and lm.supports_reasoning and lm.supports_response_schema
    assert {"reasoning_effort", "max_tokens", "response_format", "tools"} <= lm.supported_params
    lm("hi")
    entry = lm.history[-1]
    assert entry["cost"] is not None and entry["cost_details"]["kind"] == "estimate"
    assert entry["cost_details"]["provider"] == "fireworks"
    assert entry["cost_details"]["metadata"]["namespaces"] == ["fireworks_ai"]
    lm.close()


def test_stated_support_fills_what_no_snapshot_says(server):
    base, _ = server
    register_provider(
        _definition(base), supports=ModelSupport(function_calling=True),
        models={"private-v1": ModelSupport(response_schema=True, function_calling=False)},
    )
    default = dspy.LM("fireworks/private-v2", api_key="k")
    assert default.supports_function_calling and not default.supports_response_schema and not default.supports_reasoning
    assert "tools" in default.supported_params
    specific = dspy.LM("fireworks/private-v1", api_key="k")
    assert not specific.supports_function_calling and specific.supports_response_schema
    assert "response_format" in specific.supported_params and "tools" not in specific.supported_params
    # A statement never prices anything.
    specific = dspy.LM("fireworks/private-v1", api_key="k", cache=False, num_retries=0)
    specific("hi")
    assert specific.history[-1]["cost"] is None
    specific.close()


def test_a_snapshot_entry_is_more_specific_than_a_statement(server):
    base, _ = server
    # The snapshot says this Fireworks model supports tools; the statement
    # says the provider does not. The per-model fact wins.
    register_provider(_definition(base), metadata_namespaces=("fireworks_ai",),
                      supports=ModelSupport(function_calling=False))
    assert dspy.LM(f"fireworks/{MODEL}", api_key="k").supports_function_calling
    assert not dspy.LM("fireworks/private-v9", api_key="k").supports_function_calling


def test_compat_object_shapes_capabilities(server):
    base, _ = server
    register_provider(_definition(base, json_schema="reject", thinking_format="none"),
                      supports=ModelSupport(function_calling=True, reasoning=True, response_schema=True))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k")
    assert lm.supports_function_calling
    assert not lm.supports_response_schema and not lm.supports_reasoning  # the wire cannot carry them
    assert "reasoning_effort" not in lm.supported_params


# ─── state ────────────────────────────────────────────────────────────


def test_state_is_the_model_string(server):
    base, _ = server
    register_provider(_definition(base))
    state = dspy.LM(f"fireworks/{MODEL}", api_key="k").dump_state()
    json.dumps(state)
    assert "engine" not in state and "api_key" not in state and "_providers" not in state
    loaded = dspy.LM.load_state(state)
    assert loaded.model == f"fireworks/{MODEL}" and select_backend(loaded).native  # bound at load time


def test_pickled_lm_keeps_its_binding(server):
    import pickle

    base, _ = server
    register_provider(_definition(base))
    lm = dspy.LM(f"fireworks/{MODEL}", api_key="k", cache=False)
    unregister_provider("fireworks")
    restored = pickle.loads(pickle.dumps(lm))
    assert select_backend(restored).native and restored._providers == lm._providers
