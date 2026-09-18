"""One execution path for native, compatibility and custom DSPy LMs."""

import asyncio
import copy
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import pydantic

from dspy._vendor.lm15.result import StreamAccumulator
from dspy._vendor.lm15.serde import request_to_dict
from dspy.clients._deprecation import warn_legacy_shortcut
from dspy.clients._http import finite_seconds
from dspy.clients.backend_selection import CLIENT_KEYS, select_backend
from dspy.clients.call_context import stream_emitted
from dspy.clients.call_result import CACHE_FORMAT, CallResult, combine, usage_dict
from dspy.clients.engines.legacy_engine import AsyncLegacyEngine, LegacyEngine
from dspy.clients.engines.lifecycle import aclosing_stream, closing_stream
from dspy.clients.engines.litellm_engine import AsyncLiteLLMEngine, LiteLLMEngine
from dspy.clients.engines.lm15_engine import AsyncLM15Engine, LM15Engine, timeouts_for
from dspy.clients.engines.stream_guard import achecked_stream, checked_stream
from dspy.clients.engines.streaming import ListenerBridge
from dspy.clients.errors import error_boundary
from dspy.clients.legacy_outputs import plain, value
from dspy.dsp.utils.settings import settings
from dspy.lm15 import (
    CacheConfig,
    LM15Error,
    Request,
    Response,
    RouterConfig,
    StreamAssemblyError,
    UnsupportedFeatureError,
    request_from_openai_chat,
)
from dspy.utils.exceptions import LMConfigurationError, LMUnsupportedFeatureError, is_retryable_lm_error
from dspy.utils.lazy_import import require

anyio = require("anyio")

IGNORED_CACHE_KEYS = ["api_key", "api_base", "base_url"]


@dataclass
class PreparedCall:
    prompt: Any
    messages: Any
    kwargs: dict
    legacy: dict
    request: Request | None
    cache: bool
    n: int
    managed: bool

    def key(self, lm, asynchronous):
        if self.request is not None:
            return {"_fn_identifier": "dspy.clients.lm15.complete.async" if asynchronous else "dspy.clients.lm15.complete",
                    "request": request_to_dict(self.request), "rollout_id": self.kwargs.get("rollout_id")}
        suffix = {"chat": "completion", "text": "text_completion", "responses": "responses_completion"}[lm.model_type]
        name = ("alitellm_" if asynchronous else "litellm_") + suffix
        key = {**self.legacy, "_fn_identifier": f"dspy.clients.lm.{name}"}
        prompt_cache = key.pop("prompt_cache", None)
        if prompt_cache is not None:
            from dspy._vendor.lm15.serde import cache_config_to_dict

            key["prompt_cache"] = cache_config_to_dict(prompt_cache)
        return key


def _check_encodable(value, where):
    """Refuse text no provider can receive before any engine is chosen.

    lm15's builder already raises the same ValueError, but only inside the
    engine attempt, where a forced native engine reports it as unexpected.
    Nested containers cover message lists, tool inputs and typed Requests.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            bad = " ".join(f"U+{ord(ch):04X}" for ch in value[exc.start:exc.end])
            raise ValueError(
                f"{where} contains text that is not valid Unicode (lone surrogate {bad}), which no "
                "provider can receive; repair the text first, e.g. text.encode('utf-8', 'replace').decode('utf-8')"
            ) from None
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_encodable(key, where)
            _check_encodable(item, where)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _check_encodable(item, where)


def prepare(lm, prompt, messages, kwargs, *, asynchronous=False, direct=False):
    kwargs = dict(kwargs)
    request = kwargs.pop("request", None)
    if isinstance(prompt, Request):
        if request is not None:
            raise TypeError("Pass a Request once, positionally or by keyword")
        request, prompt = prompt, None
    managed = hasattr(lm, "_engine_spec")
    if managed and not direct:
        from dspy.clients.lm import LM
        from dspy.utils.dummies import DummyLM

        base = DummyLM if isinstance(lm, DummyLM) else LM if isinstance(lm, LM) else None
        if base is not None:
            method = "aforward" if asynchronous else "forward"
            managed = method not in vars(lm) and getattr(type(lm), method) is getattr(base, method)
            if base is DummyLM and asynchronous:
                # DummyLM's inherited aforward delegates to forward, including
                # a subclass's override. Do not bypass that override either.
                managed = managed and "forward" not in vars(lm) and type(lm).forward is DummyLM.forward
    if getattr(type(lm), "forward_contract", "legacy") != "legacy":
        raise TypeError("The DSPy 3.3 typed_lm contract was removed; implement an lm15 engine instead.")
    if request is not None:
        if not isinstance(request, Request):
            raise TypeError("request must be a dspy.lm15.Request")
        if prompt is not None or messages is not None:
            raise TypeError("Do not combine a Request with prompt/messages")
        if request.model != lm.model:
            raise ValueError("Request.model must match LM.model")
        from dspy.clients.engines.base import validate_request

        validate_request(request)
        _check_encodable(request_to_dict(request), "request")
        extra = set(kwargs) - {"cache", "rollout_id"}
        if extra:
            raise TypeError(f"Generation options belong in Request.config: {sorted(extra)}")
        from dspy.clients.lm15_boundary import snapshot_request

        request = snapshot_request(request)
        legacy = {"model": lm.model}
        use_cache = kwargs.get("cache", lm.cache) if managed and getattr(lm, "_cache_responses", True) else False
        return PreparedCall(None, None, kwargs, legacy, request, use_cache, 1, managed)
    if managed and not isinstance(getattr(lm, "_engine_spec", "auto"), str):
        # A custom engine owns its connection; a call cannot hand it one either.
        from dspy.clients.lm import _refuse_client_settings

        _refuse_client_settings(lm._engine_spec, kwargs, where="LM call")
    merged = {**lm.kwargs, **{key: val for key, val in kwargs.items() if key != "cache"}}
    prompt_cache = merged.get("prompt_cache")
    if prompt_cache is not None:
        if not isinstance(prompt_cache, CacheConfig):
            raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
        if not managed or getattr(lm, "_engine_spec", None) == "litellm" or lm.model_type == "text":
            raise LMUnsupportedFeatureError("prompt_cache requires native lm15 or a canonical custom engine.")
    if merged.get("rollout_id") is None:
        merged.pop("rollout_id", None)
    if hasattr(lm, "_warn_zero_temp_rollout"):
        lm._warn_zero_temp_rollout(merged.get("temperature"), merged.get("rollout_id"))
    rendered = messages or [{"role": "user", "content": prompt}]
    _check_encodable(rendered, "messages" if messages else "prompt")
    if getattr(lm, "use_developer_role", False) and lm.model_type == "responses":
        rendered = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in rendered]
    n = merged.get("n", 1)
    if n is None:
        n = 1
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    use_cache = kwargs.get("cache", lm.cache) if managed and getattr(lm, "_cache_responses", True) else False
    return PreparedCall(prompt, messages, kwargs, {"model": lm.model, "messages": rendered, **merged},
                        None, use_cache, n, managed)


def _canonical(call, *, compat=None):
    if call.request is not None:
        return call.request
    body = {key: val for key, val in call.legacy.items()
            if key not in CLIENT_KEYS | {"n", "rollout_id", "num_generations", "prompt_cache"} and val is not None}
    format_ = body.get("response_format")
    if isinstance(format_, type) and issubclass(format_, pydantic.BaseModel):
        from dspy.clients.legacy_requests import _strict_json_schema

        # A generated schema is sent strict, so it is shaped as strict mode
        # takes it. Raw caller-supplied schemas remain unchanged.
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": format_.__name__, "schema": _strict_json_schema(format_.model_json_schema()), "strict": True,
        }}
    request = request_from_openai_chat(body, compat=compat)
    prompt_cache = call.legacy.get("prompt_cache")
    if prompt_cache is not None:
        if request.config.cache is not None:
            raise LMUnsupportedFeatureError("Do not combine prompt_cache with provider-shaped prompt-cache options.")
        request = replace(request, config=replace(request.config, cache=prompt_cache))
    return request


# LiteLLM's generic doors, by lm15 dialect: the route a declared provider
# takes when a call falls back, with the declared address and credential
# carried along. The provider's own aliases are never handed to LiteLLM as
# a prefix: LiteLLM may know that name as a different service. Each door
# sends the credential under one scheme (the OpenAI SDK's bearer header,
# the Anthropic SDK's x-api-key header) — the only scheme it can carry.
_LITELLM_GENERIC_DOOR = {
    "openai-chat": ("openai", "bearer"),
    "openai-responses": ("openai", "bearer"),
    "anthropic": ("anthropic", "x-api-key"),
}


def _declared_fallback(lm, binding, resolution, clients):
    """(wire model, client options) for LiteLLM to reach a declared provider
    at its declared address with its own credential, sent under the scheme
    the declaration selects for it — never LiteLLM's idea of a similarly
    named service, never an ambient OPENAI_API_KEY, never a scheme the
    declaration did not name."""
    from dspy._vendor.lm15.access import select_scheme
    from dspy._vendor.lm15.credentials import AwsCredentials
    from dspy._vendor.lm15.providers.base import resolve_credential_value

    definition = binding.definition
    door, sends = _LITELLM_GENERIC_DOOR.get(definition.dialect, (None, None))
    if door is None or definition.hosted:
        raise LMUnsupportedFeatureError(
            f"{lm.model!r}: the declared provider {definition.id!r} has no LiteLLM route for these client "
            f"settings; use settings the native engine carries (api_key, api_base, timeout)",
            model=lm.model, provider=definition.id,
        )
    options = dict(clients)
    if not (options.get("api_base") or options.get("base_url")):
        options["api_base"] = definition.access.base_url
    key = options.get("api_key")
    if not key:
        key = next((os.environ[name] for name in definition.access.env_keys if os.environ.get(name)), None)
        key = key or definition.placeholder_key
        if not key:
            variables = " or ".join(definition.access.env_keys) or "api_key="
            raise LMConfigurationError(
                f"no credential for the declared provider {definition.id!r}: set {variables} or pass api_key=",
                model=lm.model, provider=definition.id,
            )
    # The scheme this credential travels under natively (lm15 AUTH-2). A
    # callable credential is invoked once per call, as the native path does.
    # The door renders exactly one scheme; a declaration that needs another
    # gets no fallback rather than its secret in a header it did not declare.
    credential = resolve_credential_value(key)
    scheme = select_scheme(definition.access, credential)
    if scheme != sends or isinstance(credential, AwsCredentials):
        raise LMUnsupportedFeatureError(
            f"{lm.model!r}: the declared provider {definition.id!r} authenticates with {scheme!r}, which the "
            f"LiteLLM {door}/ door cannot send (it sends {sends!r}); use settings the native engine carries "
            "(api_key, api_base, timeout)",
            model=lm.model, provider=definition.id, features=[scheme],
        )
    options["api_key"] = credential.value  # LiteLLM takes the string
    static = dict(definition.access.headers)
    if static:
        options["extra_headers"] = {**static, **(options.get("extra_headers") or {})}
    return f"{door}/{resolution.model}", options


def _engine(lm, call, asynchronous):
    # Setup is outside the retry loop. Canonical routing/conversion refusals
    # still need the same public error projection as engine-call failures.
    with error_boundary(lm.model):
        return _select_engine(lm, call, asynchronous)


def _select_engine(lm, call, asynchronous):
    if not call.managed:
        backend = AsyncLegacyEngine(lm, _implicit=True) if asynchronous else LegacyEngine(lm, _implicit=True)
        return backend, _canonical(call) if call.request else None, None
    spec = lm._engine_spec
    if not isinstance(spec, str):
        backend = lm._async_engine_spec if asynchronous else spec
        if backend is None:
            raise LMUnsupportedFeatureError("This custom engine has no async counterpart; pass async_engine=.")
        # TODO(3.5): remove the legacy shortcut after moving DummyLM and all
        # adapter execution to canonical requests/responses. Do not warn users
        # about DSPy's own temporary implementation, or repeat wrapper warnings.
        legacy_complete = getattr(backend, "complete_legacy", None)
        if call.request is None and call.legacy.get("prompt_cache") is None and callable(legacy_complete):
            from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine

            builtin_methods = (
                DummyEngine.complete_legacy, AsyncDummyEngine.complete_legacy,
                LiteLLMEngine.complete_legacy, AsyncLiteLLMEngine.complete_legacy,
            )
            implementation = getattr(legacy_complete, "__func__", legacy_complete)
            if not isinstance(backend, (LegacyEngine, AsyncLegacyEngine)) and not any(
                implementation is method for method in builtin_methods
            ):
                warn_legacy_shortcut()
            return backend, None, None
        return backend, _canonical(call), None
    selection = select_backend(lm, call.legacy)
    native, resolution, clients = selection.native, selection.resolution, selection.clients
    from dspy.lm15 import _binding_for, _definitions

    binding = _binding_for(lm._providers, resolution.provider) if resolution is not None and resolution.declared else None
    canonical = call.request
    if native:
        try:
            canonical = _canonical(call, compat=resolution.compat)
        except (LM15Error, TypeError, ValueError) as exc:
            # A declared provider's compat is the caller's own statement of
            # what that server takes; a refusal it produces is final, not a
            # reason to send the request through LiteLLM anyway.
            if (spec == "lm15" or binding is not None or call.request is not None
                    or call.legacy.get("prompt_cache") is not None):
                if isinstance(exc, UnsupportedFeatureError):
                    raise  # The outer boundary preserves its feature and other diagnostics.
                raise LMUnsupportedFeatureError(str(exc), model=lm.model) from exc
            # This is a representational refusal BEFORE execution. Preserve the
            # original body on the compatibility backend, never retry elsewhere.
            native = False

    def compatibility_engine():
        if call.legacy.get("prompt_cache") is not None:
            raise LMUnsupportedFeatureError(
                "This request requires LiteLLM, which has no prompt_cache bridge. "
                "Use native-compatible inputs/settings or omit prompt_cache."
            )
        # None disables the bridge; it is never a provider keyword.
        call.legacy.pop("prompt_cache", None)
        cls = AsyncLiteLLMEngine if asynchronous else LiteLLMEngine
        if binding is not None:
            # A declared provider keeps its destination and credential across
            # the backend switch; only the LM's model string stays as it was
            # (history, cache key). It is still that provider: errors and
            # pricing name it and read its metadata namespaces.
            wire_model, options = _declared_fallback(lm, binding, resolution, clients)
            engine = cls(model_type=lm.model_type, wire_model=wire_model, **options)
            return engine, canonical if call.request else None, resolution.provider
        return cls(model_type=lm.model_type, **clients), canonical if call.request else None, None

    if not native:
        return compatibility_engine()
    # Long-lived sync pools and a separate async pool per event loop. Copies
    # share the store; a copy with changed client settings gets a distinct key.
    loop = asyncio.get_running_loop() if asynchronous else None
    timeouts = timeouts_for(clients.get("timeout"))
    key = (loop, lm.model, lm.model_type, timeouts,
           tuple(sorted((k, v if isinstance(v, (str, int, float, type(None))) else id(v))
                        for k, v in clients.items() if k != "timeout")))
    with lm._engine_lock:
        backend = lm._engine_store.get(key)
        if backend is None:
            if loop is not None:
                lm._reap_closed_loops()
            provider = resolution.provider
            api_keys = {provider: clients["api_key"]} if "api_key" in clients else None
            url = clients.get("api_base") or clients.get("base_url")
            config = RouterConfig(api_keys=api_keys, base_urls={provider: url} if url else None, timeouts=timeouts,
                                  providers=_definitions(lm._providers))
            cls = AsyncLM15Engine if asynchronous else LM15Engine
            backend = cls(config, model_type=lm.model_type)
            lm._engine_store[key] = backend
    if spec == "auto" and canonical is not None and call.legacy.get("prompt_cache") is None:
        # The migration promise: an input the native route cannot carry
        # selects LiteLLM BEFORE any I/O. lm15 MAP-13 adapts most settings
        # and records it; what it still refuses is known from plan() with
        # no network, so the fallback happens here, not after a failed call.
        try:
            backend.plan(canonical)
        except UnsupportedFeatureError:
            if binding is not None:
                # The declaration is the authority on what that server cannot
                # do; sending the request through LiteLLM anyway would defeat
                # the refusal it encodes (a silent degrade). Refuse.
                raise
            return compatibility_engine()
    return backend, canonical, resolution.provider


def _cached(lm, call, asynchronous):
    if not call.cache:
        return None
    import dspy

    record = dspy.cache.get(call.key(lm, asynchronous), IGNORED_CACHE_KEYS)
    if record is None:
        return None
    if isinstance(record, dict) and record.get("_dspy_format") == CACHE_FORMAT:
        return CallResult.load(record)
    result = CallResult.legacy(lm, record, kwargs=call.legacy)
    result.cache_hit = True
    result.usage = {}
    return result


def _store(lm, call, result, asynchronous):
    if call.cache:
        import dspy

        # Preserve the original SDK cache format on the compatibility path.
        # Native responses use only plain data, compatible with restricted mode.
        record = result.dump() if result.responses or result.raw is None else result.raw
        dspy.cache.put(call.key(lm, asynchronous), record, IGNORED_CACHE_KEYS)


def _delay(exc, attempt):
    hint = finite_seconds(getattr(exc, "retry_after", None))
    return hint if hint is not None else min(2 ** min(attempt, 6), 60)


def _result(lm, response, call, provider, request=None, *, estimate=True):
    if not isinstance(response, Response):
        actual = f"{type(response).__module__}.{type(response).__qualname__}"
        raise TypeError(f"Engine.complete must return dspy.lm15.Response, got {actual}")
    result = CallResult.native(response, model_type=lm.model_type,
                               logprobs=(call.request.config.logprobs is not None if call.request else bool(call.legacy.get("logprobs"))),
                               provider=provider)
    if estimate:
        _price_result(lm, result, response, provider, request)
    if response.finish_reason == "length":
        import logging

        logging.getLogger("dspy.clients.lm").warning("LM response was truncated; increase max_tokens or inspect history.")
    return result


def _price_legacy(lm, result, provider):
    """A declared provider reached through LiteLLM's generic door is still
    that provider: its usage is priced from the declaration's namespaces,
    replacing LiteLLM's reading of the door's model string (a colliding
    name would be priced as another vendor's; an unknown one not at all).
    Chat only; other model types keep LiteLLM's figure."""
    if provider is None or result.raw is None or lm.model_type != "chat":
        return
    try:
        from dspy._vendor.lm15.providers.openai_chat import response_from_openai_chat

        response = response_from_openai_chat(plain(result.raw), model=lm.model)
    except Exception:
        return  # advisory: never fail a completed call over pricing
    _price_result(lm, result, response, provider, None)


def _price_result(lm, result, response, provider, request):
    """Advisory pricing only: safe to finish in a worker after cancellation.

    Never update usage, history, or the response cache from this worker.
    """
    from dspy.clients.costs import estimate_cost

    try:
        wire_model = request.model if request is not None else lm.model
        namespaces = None
        if provider is not None:
            from dspy.clients.capabilities import resolve
            from dspy.lm15 import _binding_for

            route = resolve(lm)
            wire_model = route.model
            binding = _binding_for(getattr(lm, "_providers", ()), provider) if route.declared else None
            if binding is not None:
                namespaces = binding.metadata_namespaces
        result.cost, result.cost_details = estimate_cost(
            response, provider=provider, requested_model=wire_model, request=request, namespaces=namespaces,
        )
    except Warning:
        raise
    except Exception:
        result.cost = None
        result.cost_details = {"kind": "unknown", "reason": "pricing metadata could not be interpreted"}


@dataclass
class _Attempt:
    emitted: bool = False
    completed: bool = False
    recorded: bool = False
    response: Response | None = None
    result: CallResult | None = None


def _retain_response_usage(state, results, response, provider):
    if isinstance(response, Response) and not state.recorded:
        # Record billing before output conversion, final delivery or cleanup.
        # If conversion fails this placeholder is accounted, never returned.
        results.append(CallResult(usage=usage_dict(response, provider)))
        state.recorded = True


def _accept_response(lm, call, state, results, response, provider, request):
    state.completed = True
    state.response = response
    _retain_response_usage(state, results, response, provider)
    state.result = _result(lm, response, call, provider, request, estimate=False)
    results[-1] = state.result


def _accept_end(lm, call, state, results, accumulator, provider, request):
    state.completed = True
    try:
        response = accumulator.response()
    except StreamAssemblyError as exc:
        _retain_response_usage(state, results, exc.partial, provider)
        raise
    _accept_response(lm, call, state, results, response, provider, request)


def _partial(accumulator):
    try:
        present = any((accumulator.started_model, accumulator.started_id, accumulator.text_parts,
                       accumulator.thinking_parts, accumulator.tool_call_meta, accumulator.image_parts,
                       accumulator.audio_chunks, accumulator.citation_parts, accumulator.message_continuation))
        return accumulator.response() if present else None
    except StreamAssemblyError as exc:
        return exc.partial
    except Exception:
        # Salvaging diagnostics must never replace the primary error.
        return None


def _legacy_done(state, results, progress):
    state.emitted = progress.get("emitted", False)
    state.completed = state.completed or progress.get("completed", False)
    if state.result is not None:
        results.append(state.result)
        state.recorded = True
    elif state.completed and "raw" in progress:
        # A completed compatibility response can outlive a failed SDK close or
        # output conversion. Extract only reported usage, not invented outputs.
        try:
            usage = plain(dict(value(progress["raw"], "usage", {}) or {}))
        except Exception:
            usage = {}
        results.append(CallResult(usage=usage))
        state.recorded = True


def _accept_legacy(state, result):
    state.completed = True
    if not isinstance(result, CallResult):
        raise TypeError("Engine.complete_legacy must return a CallResult")
    state.result = result


def _hint_responses_api(lm, exc):
    """OpenAI serves function tools for its reasoning models on the Responses
    API only, and says so in its refusal ("use /v1/responses"). DSPy has the
    switch — model_type="responses" — so the error names it. No endpoint is
    chosen for the caller: which models need it is OpenAI's policy, not
    DSPy's to guess."""
    from dspy.utils.exceptions import LMInvalidRequestError

    if not isinstance(exc, LMInvalidRequestError) or getattr(lm, "model_type", None) != "chat":
        return
    message = getattr(exc, "message", "") or ""
    if "/v1/responses" not in message or "DSPy:" in message:
        return
    hint = " DSPy: construct the LM with model_type='responses' to use the Responses API for this model."
    exc.message = message + hint
    if exc.args and isinstance(exc.args[0], str):
        exc.args = (exc.args[0] + hint, *exc.args[1:])


def _account_failed_call(lm, results, primary):
    if settings.usage_tracker:
        for result in results:
            try:
                settings.usage_tracker.add_usage(lm.model, result.usage)
            except Exception as secondary:
                # A custom tracker must not replace an engine error or cancel.
                try:
                    primary.usage_errors = (*getattr(primary, "usage_errors", ()), secondary)
                    if hasattr(primary, "add_note"):
                        primary.add_note(f"Usage accounting also failed ({type(secondary).__name__}); see usage_errors.")
                except Exception:
                    pass


def _retryable(state, exc, attempt, retries):
    return not state.completed and not state.emitted and attempt < retries and is_retryable_lm_error(exc)


def _attempt(lm, call, backend, request, provider, state, results):
    stream = settings.send_stream
    if request is None:
        progress = {"emitted": False}
        try:
            with settings.context(_lm_stream_progress=progress):
                result = backend.complete_legacy(lm, copy.deepcopy(call.legacy), prompt=call.prompt,
                                                 messages=call.messages, call_kwargs=call.kwargs)
                _accept_legacy(state, result)
                _price_legacy(lm, result, provider)
        finally:
            _legacy_done(state, results, progress)
    elif stream is None:
        try:
            response = backend.complete(request)
        except StreamAssemblyError as exc:
            _retain_response_usage(state, results, exc.partial, provider)
            raise
        _accept_response(lm, call, state, results, response, provider, request)
    else:
        accumulator = StreamAccumulator(request)
        bridge = ListenerBridge(lm.model, id(settings.caller_predict) if settings.caller_predict else None)
        events = checked_stream(backend.stream(request), provider=provider)
        try:
            with closing_stream(events):
                for event in events:
                    accumulator.push(event)
                    if event.type == "end":
                        _accept_end(lm, call, state, results, accumulator, provider, request)
                    if chunk := bridge.chunk(event):
                        state.emitted = True
                        stream_emitted()
                        anyio.from_thread.run(stream.send, chunk)
        except StreamAssemblyError as exc:
            if exc.partial is None:
                exc.partial = _partial(accumulator)
            raise


def execute(lm, call):
    if result := _cached(lm, call, False):
        return result
    backend, request, provider = _engine(lm, call, False)
    results = []
    # Ordinary compatibility calls keep backend-native n; canonical engines
    # produce one candidate per attempt.
    # TODO(candidate-parallelism): bound concurrent canonical requests while
    # respecting engine concurrency guarantees, output order, per-candidate
    # retries/cancellation accounting, and whole-call caching. Define candidate
    # identity for listeners before multiplexing streams. Concurrency reduces
    # latency, not the input-token charges for separate requests.
    count = call.n if request is not None else 1
    retries = lm.num_retries if call.managed else 0
    try:
        for _ in range(count):
            for attempt in range(retries + 1):
                state = _Attempt()
                try:
                    boundary = error_boundary(lm.model, provider=provider, unexpected=True) if call.managed else nullcontext()
                    with boundary:
                        _attempt(lm, call, backend, request, provider, state, results)
                    break
                except Exception as exc:
                    if not _retryable(state, exc, attempt, retries):
                        raise
                    time.sleep(_delay(exc, attempt))
            # Completion is irreversible. Advisory pricing and storage never
            # run inside the retry region, even if custom code raises here.
            if request is not None:
                _price_result(lm, state.result, state.response, provider, request)
        result = results[0] if len(results) == 1 else combine(results, model_type=lm.model_type)
        _store(lm, call, result, False)
        return result
    except BaseException as exc:
        _hint_responses_api(lm, exc)
        _account_failed_call(lm, results, exc)
        raise


async def _aattempt(lm, call, backend, request, provider, state, results):
    stream = settings.send_stream
    if request is None:
        progress = {"emitted": False}
        try:
            with settings.context(_lm_stream_progress=progress):
                result = await backend.complete_legacy(lm, copy.deepcopy(call.legacy), prompt=call.prompt,
                                                       messages=call.messages, call_kwargs=call.kwargs)
                _accept_legacy(state, result)
                await asyncio.to_thread(_price_legacy, lm, result, provider)  # metadata lookup off the loop
        finally:
            _legacy_done(state, results, progress)
    elif stream is None:
        try:
            response = await backend.complete(request)
        except StreamAssemblyError as exc:
            _retain_response_usage(state, results, exc.partial, provider)
            raise
        _accept_response(lm, call, state, results, response, provider, request)
    else:
        accumulator = StreamAccumulator(request)
        bridge = ListenerBridge(lm.model, id(settings.caller_predict) if settings.caller_predict else None)
        events = achecked_stream(backend.stream(request), provider=provider)
        try:
            async with aclosing_stream(events):
                async for event in events:
                    accumulator.push(event)
                    if event.type == "end":
                        _accept_end(lm, call, state, results, accumulator, provider, request)
                    if chunk := bridge.chunk(event):
                        state.emitted = True
                        stream_emitted()
                        await stream.send(chunk)
        except StreamAssemblyError as exc:
            if exc.partial is None:
                exc.partial = _partial(accumulator)
            raise


async def aexecute(lm, call):
    if call.cache:
        if result := await asyncio.to_thread(_cached, lm, call, True):
            return result
    backend, request, provider = _engine(lm, call, True)
    results = []
    # TODO(candidate-parallelism): apply the same guarantees as execute() above;
    # cancellation must account for every completed candidate exactly once.
    count = call.n if request is not None else 1
    retries = lm.num_retries if call.managed else 0
    try:
        for _ in range(count):
            for attempt in range(retries + 1):
                state = _Attempt()
                try:
                    boundary = error_boundary(lm.model, provider=provider, unexpected=True) if call.managed else nullcontext()
                    with boundary:
                        await _aattempt(lm, call, backend, request, provider, state, results)
                    break
                except Exception as exc:
                    if not _retryable(state, exc, attempt, retries):
                        raise
                    await asyncio.sleep(_delay(exc, attempt))
            if request is not None:
                await asyncio.to_thread(_price_result, lm, state.result, state.response, provider, request)
        result = results[0] if len(results) == 1 else combine(results, model_type=lm.model_type)
        if call.cache:
            # Cancellation may leave this worker writing a complete result,
            # never a partial one. Usage still belongs to this public call.
            await asyncio.to_thread(_store, lm, call, result, True)
        return result
    except BaseException as exc:
        _hint_responses_api(lm, exc)
        _account_failed_call(lm, results, exc)
        raise


def finalize(lm, call, result):
    if not result.cache_hit and settings.usage_tracker:
        settings.usage_tracker.add_usage(lm.model, result.usage)
    if not settings.disable_history:
        import datetime
        import uuid

        from dspy.clients.lm15_boundary import history_messages

        entry = {"prompt": call.prompt, "messages": history_messages(call.request) if call.request else call.messages,
                 "kwargs": {key: val for key, val in call.kwargs.items() if not key.startswith("api_")},
                 "response": result.raw if result.raw is not None else result.responses,
                 "outputs": result.outputs, "usage": result.usage, "cost": result.cost,
                 "timestamp": datetime.datetime.now().isoformat(), "uuid": str(uuid.uuid4()),
                 "model": lm.model, "response_model": result.response_model, "model_type": lm.model_type}
        if result.cost_details:
            entry["cost_details"] = result.cost_details
        if call.request:
            entry["request"] = call.request
        # lm15 MAP-13: what the wire got that differs from what was asked
        # (a dropped seed, a clamped temperature). Data on the entry, never
        # printed; absent when the request went out as written.
        adaptations = tuple(a for response in result.responses for a in getattr(response, "adaptations", ()))
        if adaptations:
            entry["adaptations"] = adaptations
        lm.update_history(entry)
    return result.typed(call.request, lm.model_type) if call.request else result.outputs
