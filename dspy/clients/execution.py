"""One execution path for native, compatibility and custom DSPy LMs."""

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any

import anyio
import pydantic

from dspy._vendor.lm15.result import StreamAccumulator
from dspy._vendor.lm15.serde import request_to_dict
from dspy.clients.call_result import CACHE_FORMAT, CallResult, combine
from dspy.clients.engines.errors import wrap_error
from dspy.clients.engines.legacy_engine import AsyncLegacyEngine, LegacyEngine
from dspy.clients.engines.litellm_engine import AsyncLiteLLMEngine, LiteLLMEngine
from dspy.clients.engines.lm15_engine import AsyncLM15Engine, LM15Engine
from dspy.clients.engines.streaming import ListenerBridge
from dspy.dsp.utils.settings import settings
from dspy.lm15 import LM15Error, Request, Response, RouterConfig, request_from_openai_chat
from dspy.utils.exceptions import LMError, LMUnsupportedFeatureError, is_retryable_lm_error

IGNORED_CACHE_KEYS = ["api_key", "api_base", "base_url"]
CLIENT_KEYS = {"api_key", "api_base", "base_url", "headers", "extra_headers", "timeout", "api_version",
               "azure_ad_token_provider", "organization", "project", "extra_query", "custom_llm_provider"}
NATIVE_CLIENT_KEYS = {"api_key", "api_base", "base_url"}


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
        return {**self.legacy, "_fn_identifier": f"dspy.clients.lm.{name}"}


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
    if not managed and not getattr(lm, "_warned_legacy_engine", False):
        import warnings

        warnings.warn(
            "Legacy BaseLM.forward plugins remain supported in DSPy 3.4. "
            "For the planned 3.5 migration, implement complete(Request) -> Response "
            "and pass it to dspy.LM(engine=...).", FutureWarning, stacklevel=4,
        )
        lm._warned_legacy_engine = True
    if getattr(type(lm), "forward_contract", "legacy") != "legacy":
        raise TypeError("The DSPy 3.3 typed_lm contract was removed; implement an lm15 engine instead.")
    if request is not None:
        if not isinstance(request, Request):
            raise TypeError("request must be a dspy.lm15.Request")
        if prompt is not None or messages is not None:
            raise TypeError("Do not combine a Request with prompt/messages")
        if request.model != lm.model:
            raise ValueError("Request.model must match LM.model")
        extra = set(kwargs) - {"cache", "rollout_id"}
        if extra:
            raise TypeError(f"Generation options belong in Request.config: {sorted(extra)}")
        from dspy.clients.lm15_boundary import snapshot_request

        request = snapshot_request(request)
        legacy = {"model": lm.model}
        use_cache = kwargs.get("cache", lm.cache) if managed and getattr(lm, "_cache_responses", True) else False
        return PreparedCall(None, None, kwargs, legacy, request, use_cache, 1, managed)
    merged = {**lm.kwargs, **{key: val for key, val in kwargs.items() if key != "cache"}}
    if merged.get("rollout_id") is None:
        merged.pop("rollout_id", None)
    if hasattr(lm, "_warn_zero_temp_rollout"):
        lm._warn_zero_temp_rollout(merged.get("temperature"), merged.get("rollout_id"))
    rendered = messages or [{"role": "user", "content": prompt}]
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
            if key not in CLIENT_KEYS | {"n", "rollout_id", "num_generations"} and val is not None}
    format_ = body.get("response_format")
    if isinstance(format_, type) and issubclass(format_, pydantic.BaseModel):
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": format_.__name__, "schema": format_.model_json_schema(), "strict": True,
        }}
    return request_from_openai_chat(body, compat=compat)


def _engine(lm, call, asynchronous):
    if not call.managed:
        backend = AsyncLegacyEngine(lm) if asynchronous else LegacyEngine(lm)
        return backend, _canonical(call) if call.request else None, None
    spec = lm._engine_spec
    if not isinstance(spec, str):
        backend = lm._async_engine_spec if asynchronous else spec
        if backend is None:
            raise LMUnsupportedFeatureError("This custom engine has no async counterpart; pass async_engine=.")
        # Built-in scripted engines can consume ordinary inputs without a
        # provider-wire decoder. Only that explicit input method opts in.
        if call.request is None and callable(getattr(backend, "complete_legacy", None)):
            return backend, None, None
        return backend, _canonical(call), None
    clients = {key: val for key, val in lm.kwargs.items() if key in CLIENT_KEYS}
    clients.update({key: val for key, val in call.legacy.items() if key in CLIENT_KEYS})
    native = spec != "litellm" and lm.model_type != "text"
    resolution = None
    if native:
        try:
            # No credential loading or network during selection.
            probe = LM15Engine(RouterConfig(env={}), model_type=lm.model_type)
            resolution = probe.resolve(lm.model)
            import os

            # Preserve legacy environment-configured gateways rather than
            # accidentally sending the same credentials to a public endpoint.
            env_prefix = lm.model.split("/", 1)[0].upper()
            if spec == "auto" and any(os.getenv(name) for name in (f"{env_prefix}_API_BASE", f"{env_prefix}_BASE_URL")):
                native = False
            if spec == "auto" and resolution.provider == "xai" and "api_key" not in clients and os.getenv("XAI_API_KEY"):
                clients["api_key"] = os.environ["XAI_API_KEY"]
            if (set(clients) - NATIVE_CLIENT_KEYS) or (resolution.provider.startswith("azure") and
                                                       any(key in clients for key in ("api_base", "base_url"))):
                native = False
        except LMError as exc:
            if spec == "lm15":
                raise
            from dspy.lm15 import UnknownModelError

            if isinstance(exc.__cause__, UnknownModelError) or isinstance(exc, LMUnsupportedFeatureError):
                native = False
            else:
                raise
    canonical = call.request
    if native:
        try:
            canonical = _canonical(call, compat=resolution.compat)
        except (LM15Error, TypeError, ValueError) as exc:
            if spec == "lm15" or call.request is not None:
                raise LMUnsupportedFeatureError(str(exc), model=lm.model) from exc
            # This is a representational refusal BEFORE execution. Preserve the
            # original body on the compatibility backend, never retry elsewhere.
            native = False
    if spec == "lm15" and not native:
        raise LMUnsupportedFeatureError("The requested client settings require the LiteLLM compatibility engine.")
    if not native:
        cls = AsyncLiteLLMEngine if asynchronous else LiteLLMEngine
        return cls(model_type=lm.model_type, **clients), canonical if call.request else None, None
    # Long-lived sync pools and a separate async pool per event loop. Copies
    # share the store; a copy with changed client settings gets a distinct key.
    loop = asyncio.get_running_loop() if asynchronous else None
    key = (loop, lm.model, lm.model_type, tuple(sorted((k, v if isinstance(v, (str, int, float, type(None))) else id(v))
                                                   for k, v in clients.items())))
    with lm._engine_lock:
        backend = lm._engine_store.get(key)
        if backend is None:
            provider = resolution.provider
            api_keys = {provider: clients["api_key"]} if "api_key" in clients else None
            url = clients.get("api_base") or clients.get("base_url")
            config = RouterConfig(api_keys=api_keys, base_urls={provider: url} if url else None)
            cls = AsyncLM15Engine if asynchronous else LM15Engine
            backend = cls(config, model_type=lm.model_type)
            lm._engine_store[key] = backend
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
    hint = getattr(exc, "retry_after", None)
    return max(float(hint), 0.0) if hint is not None else min(2 ** attempt, 60)


def _result(lm, response, call, provider):
    if not isinstance(response, Response):
        raise TypeError(f"Engine.complete must return dspy.lm15.Response, got {type(response).__name__}")
    result = CallResult.native(response, model_type=lm.model_type,
                               logprobs=(call.request.config.logprobs is not None if call.request else bool(call.legacy.get("logprobs"))),
                               provider=provider)
    if response.finish_reason == "length":
        import logging

        logging.getLogger("dspy.clients.lm").warning("LM response was truncated; increase max_tokens or inspect history.")
    return result


def execute(lm, call):
    if result := _cached(lm, call, False):
        return result
    backend, request, provider = _engine(lm, call, False)
    stream = settings.send_stream
    results = []
    # LiteLLM and legacy plugins keep native n behavior when using ordinary
    # inputs. Canonical engines have one candidate per attempt.
    count = call.n if request is not None else 1
    for _ in range(count):
        emitted = False
        retries = lm.num_retries if call.managed else 0
        for attempt in range(retries + 1):
            try:
                if request is None:
                    # The compatibility streaming driver sets this flag before
                    # sending any chunk; a partial stream must never be replayed.
                    progress = {"emitted": False}
                    with settings.context(_lm_stream_progress=progress):
                        try:
                            result = backend.complete_legacy(lm, copy.deepcopy(call.legacy), prompt=call.prompt,
                                                             messages=call.messages, call_kwargs=call.kwargs)
                        finally:
                            emitted = progress["emitted"]
                elif stream is None:
                    result = _result(lm, backend.complete(request), call, provider)
                else:
                    accumulator = StreamAccumulator(request)
                    bridge = ListenerBridge(lm.model, id(settings.caller_predict) if settings.caller_predict else None)
                    source = backend.stream(request)
                    try:
                        for event in source:
                            accumulator.push(event)
                            if chunk := bridge.chunk(event):
                                emitted = True
                                anyio.from_thread.run(stream.send, chunk)
                    finally:
                        close = getattr(source, "close", None)
                        if close:
                            close()
                    result = _result(lm, accumulator.response(), call, provider)
                results.append(result)
                break
            except Exception as exc:
                error = wrap_error(exc, model=lm.model, provider=provider) if call.managed else exc
                if not emitted and attempt < retries and is_retryable_lm_error(error):
                    time.sleep(_delay(error, attempt))
                    continue
                # Earlier candidates really consumed tokens, even if a later
                # candidate fails. Do not cache a partially completed n call.
                if settings.usage_tracker:
                    for completed in results:
                        settings.usage_tracker.add_usage(lm.model, completed.usage)
                if error is exc:
                    raise
                raise error from exc
    result = results[0] if len(results) == 1 else combine(results, model_type=lm.model_type)
    _store(lm, call, result, False)
    return result


async def aexecute(lm, call):
    if result := _cached(lm, call, True):
        return result
    backend, request, provider = _engine(lm, call, True)
    stream = settings.send_stream
    results = []
    count = call.n if request is not None else 1
    for _ in range(count):
        emitted = False
        retries = lm.num_retries if call.managed else 0
        for attempt in range(retries + 1):
            try:
                if request is None:
                    progress = {"emitted": False}
                    with settings.context(_lm_stream_progress=progress):
                        try:
                            result = await backend.complete_legacy(lm, copy.deepcopy(call.legacy), prompt=call.prompt,
                                                                   messages=call.messages, call_kwargs=call.kwargs)
                        finally:
                            emitted = progress["emitted"]
                elif stream is None:
                    result = _result(lm, await backend.complete(request), call, provider)
                else:
                    accumulator = StreamAccumulator(request)
                    bridge = ListenerBridge(lm.model, id(settings.caller_predict) if settings.caller_predict else None)
                    source = backend.stream(request)
                    try:
                        async for event in source:
                            accumulator.push(event)
                            if chunk := bridge.chunk(event):
                                emitted = True
                                await stream.send(chunk)
                    finally:
                        close = getattr(source, "aclose", None)
                        if close:
                            with anyio.CancelScope(shield=True):
                                await close()
                    result = _result(lm, accumulator.response(), call, provider)
                results.append(result)
                break
            except Exception as exc:
                error = wrap_error(exc, model=lm.model, provider=provider) if call.managed else exc
                if not emitted and attempt < retries and is_retryable_lm_error(error):
                    await asyncio.sleep(_delay(error, attempt))
                    continue
                if settings.usage_tracker:
                    for completed in results:
                        settings.usage_tracker.add_usage(lm.model, completed.usage)
                if error is exc:
                    raise
                raise error from exc
    result = results[0] if len(results) == 1 else combine(results, model_type=lm.model_type)
    _store(lm, call, result, True)
    return result


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
        if call.request:
            entry["request"] = call.request
        lm.update_history(entry)
    return result.typed(call.request, lm.model_type) if call.request else result.outputs
