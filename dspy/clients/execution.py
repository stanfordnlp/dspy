"""One execution path for native and custom DSPy LMs.

Every call becomes one lm15 Request before anything here runs. This module
owns what engines do not: cache reads and writes, retries, candidate fan-out,
usage accounting, history and the public return shapes.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from dspy._vendor.lm15.result import StreamAccumulator
from dspy._vendor.lm15.serde import request_to_dict
from dspy.clients._http import finite_seconds
from dspy.clients.backend_selection import select_backend
from dspy.clients.call_context import stream_emitted
from dspy.clients.call_result import CACHE_FORMAT, CallResult, combine, usage_dict
from dspy.clients.engines.base import validate_request
from dspy.clients.engines.lifecycle import aclosing_stream, closing_stream
from dspy.clients.engines.lm15_engine import AsyncLM15Engine, LM15Engine, timeouts_for
from dspy.clients.engines.stream_guard import achecked_stream, checked_stream
from dspy.clients.engines.streaming import ListenerBridge
from dspy.clients.errors import error_boundary
from dspy.clients.lm15_boundary import snapshot_request
from dspy.clients.requests import CLIENT_KEYS, build_request
from dspy.dsp.utils.settings import settings
from dspy.lm15 import ProviderError, Request, Response, RouterConfig, StreamAssemblyError
from dspy.utils.exceptions import LMUnsupportedFeatureError, is_retryable_lm_error

IGNORED_CACHE_KEYS = ["api_key", "api_base", "base_url"]
_MESSAGES_REMOVED = (
    "lm(messages=[...]) was removed in DSPy 3.5. Build a dspy.lm15.Request with Message objects "
    "(system text goes in Request.system) and call lm(request); it returns an lm15 Response. "
    "See https://dspy.ai/community/normalized-lm-api-migration/#migrating-openai-style-messages."
)


@dataclass
class PreparedCall:
    request: Request
    prompt: str | None
    options: dict
    cache: bool
    rollout_id: Any
    n: int
    convenience: bool
    asynchronous: bool

    def key(self, lm):
        key = {"_fn_identifier": "dspy.clients.lm15.complete.async" if self.asynchronous else "dspy.clients.lm15.complete",
               "request": request_to_dict(self.request),
               "model_type": lm.model_type, "rollout_id": self.rollout_id,
               **{key: value for key, value in {**lm.kwargs, **self.options}.items() if key in CLIENT_KEYS}}
        if self.n != 1:
            key["n"] = self.n
        return key


def _check_encodable(value, where):
    """Refuse text no provider can receive before any engine is chosen."""
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


def _count(value, lm):
    if value is None:
        value = lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("n must be a positive integer")
    return value


def prepare(lm, prompt, options, *, cache=None, rollout_id=None, asynchronous=False):
    options = dict(options)
    if not isinstance(lm.engine, str):
        from dspy.clients.lm import _refuse_client_settings

        _refuse_client_settings(lm.engine, options, where="LM call")
    if "messages" in options:
        raise TypeError(_MESSAGES_REMOVED)
    if "request" in options:
        if prompt is not None:
            raise TypeError("Pass a Request once, positionally or by keyword")
        prompt = options.pop("request")
    use_cache = (lm.cache if cache is None else cache) and getattr(lm, "_cache_responses", True)
    if rollout_id is None:
        rollout_id = lm.kwargs.get("rollout_id")
    if isinstance(prompt, Request):
        if set(options) & {"n", "num_generations"}:
            raise TypeError("lm(Request) returns one Response; use Predict(n=...) for multiple candidates")
        extra = set(options) - CLIENT_KEYS
        if extra:
            raise TypeError(f"Generation options belong in Request.config: {sorted(extra)}")
        if prompt.model != lm.model:
            raise ValueError("Request.model must match LM.model")
        validate_request(prompt)
        _check_encodable(request_to_dict(prompt), "request")
        return PreparedCall(snapshot_request(prompt), None, options, bool(use_cache), rollout_id, 1, False, asynchronous)
    if not isinstance(prompt, str):
        raise TypeError("An LM call takes a dspy.lm15.Request or a prompt string")
    count = _count(options.pop("n", options.pop("num_generations", None)), lm)
    if hasattr(lm, "_warn_zero_temp_rollout"):
        lm._warn_zero_temp_rollout({**lm.kwargs, **options}.get("temperature"), rollout_id)
    _check_encodable(prompt, "prompt")
    request = snapshot_request(build_request(lm, prompt, options))
    return PreparedCall(request, prompt, options, bool(use_cache), rollout_id, count, True, asynchronous)


def _engine(lm, call, asynchronous):
    # Setup is outside the retry loop. Routing refusals need the same public
    # error projection as engine-call failures.
    with error_boundary(lm.model):
        return _select_engine(lm, call, asynchronous)


def _select_engine(lm, call, asynchronous):
    spec = lm.engine
    if not isinstance(spec, str):
        backend = lm.async_engine if asynchronous else spec
        if backend is None:
            what = "async engine; pass async_engine= with `async complete(Request)`" if asynchronous else (
                "engine; pass engine= with `complete(Request) -> Response`")
            raise LMUnsupportedFeatureError(
                f"{type(lm).__name__} has no {what}. Custom LMs are engines in DSPy 3.5; "
                "see https://dspy.ai/community/normalized-lm-api-migration/.", model=lm.model,
            )
        return backend, None
    selection = select_backend(lm, call.options, request=call.request)
    resolution, clients = selection.resolution, selection.clients
    from dspy.lm15 import _definitions

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
    return backend, resolution.provider


def _cached(lm, call):
    if not call.cache:
        return None
    import dspy

    record = dspy.cache.get(call.key(lm), IGNORED_CACHE_KEYS)
    if isinstance(record, dict) and record.get("_dspy_format") == CACHE_FORMAT:
        return CallResult.load(record)
    return None


def _store(lm, call, result):
    if call.cache:
        import dspy

        dspy.cache.put(call.key(lm), result.dump(), IGNORED_CACHE_KEYS)


def _delay(exc, attempt):
    hint = finite_seconds(getattr(exc, "retry_after", None))
    return hint if hint is not None else min(2 ** min(attempt, 6), 60)


def _result(lm, response, call, provider):
    if not isinstance(response, Response):
        actual = f"{type(response).__module__}.{type(response).__qualname__}"
        raise TypeError(f"Engine.complete must return dspy.lm15.Response, got {actual}")
    result = CallResult.native(response, model_type=lm.model_type,
                               logprobs=call.request.config.logprobs is not None, provider=provider)
    if response.finish_reason == "length":
        import logging

        logging.getLogger("dspy.clients.lm").warning(
            "LM response was truncated; increase max_tokens or inspect the response with `dspy.inspect_history()`."
        )
    return result


def _price_result(lm, result, response, provider, request):
    """Advisory pricing only: safe to finish in a worker after cancellation.

    Never update usage, history, or the response cache from this worker.
    """
    from dspy.clients.costs import estimate_cost

    try:
        wire_model = request.model
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


def _accept_response(lm, call, state, results, response, provider):
    state.completed = True
    state.response = response
    _retain_response_usage(state, results, response, provider)
    state.result = _result(lm, response, call, provider)
    results[-1] = state.result


def _accept_end(lm, call, state, results, accumulator, provider):
    from dspy.clients.lm15_boundary import repair_tool_arguments

    state.completed = True
    try:
        response = accumulator.response()
    except StreamAssemblyError as exc:
        _retain_response_usage(state, results, exc.partial, provider)
        raise
    response = repair_tool_arguments(response, (
        (str(meta.get("id") or f"tool_call_{index}"), str(meta["name"]), raw)
        for index, raw in accumulator.tool_call_raw.items()
        if (meta := accumulator.tool_call_meta.get(index)) and meta.get("name")
    ))
    _accept_response(lm, call, state, results, response, provider)


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


def _hint_responses_api(lm, exc):
    """Name DSPy's endpoint switch when the provider requires Responses."""
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
    _hint_responses_api(lm, primary)
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


def _bridge(lm):
    return ListenerBridge(lm.model, id(settings.caller_predict) if settings.caller_predict else None)


def _attempt(lm, call, backend, provider, state, results):
    stream = settings.send_stream
    request = call.request
    if stream is None:
        try:
            response = backend.complete(request)
        except (ProviderError, StreamAssemblyError) as exc:
            _retain_response_usage(state, results, getattr(exc, "partial", None), provider)
            raise
        _accept_response(lm, call, state, results, response, provider)
        return
    accumulator = StreamAccumulator(request)
    bridge = _bridge(lm)
    events = checked_stream(backend.stream(request), provider=provider)
    try:
        with closing_stream(events):
            for event in events:
                accumulator.push(event)
                if event.type == "end":
                    _accept_end(lm, call, state, results, accumulator, provider)
                if chunk := bridge.chunk(event):
                    state.emitted = True
                    stream_emitted()
                    from dspy.utils.lazy_import import require

                    require("anyio").from_thread.run(stream.send, chunk)
    except StreamAssemblyError as exc:
        if exc.partial is None:
            exc.partial = _partial(accumulator)
        raise


async def _aattempt(lm, call, backend, provider, state, results):
    stream = settings.send_stream
    request = call.request
    if stream is None:
        try:
            response = await backend.complete(request)
        except (ProviderError, StreamAssemblyError) as exc:
            _retain_response_usage(state, results, getattr(exc, "partial", None), provider)
            raise
        _accept_response(lm, call, state, results, response, provider)
        return
    accumulator = StreamAccumulator(request)
    bridge = _bridge(lm)
    events = achecked_stream(backend.stream(request), provider=provider)
    try:
        async with aclosing_stream(events):
            async for event in events:
                accumulator.push(event)
                if event.type == "end":
                    _accept_end(lm, call, state, results, accumulator, provider)
                if chunk := bridge.chunk(event):
                    state.emitted = True
                    stream_emitted()
                    await stream.send(chunk)
    except StreamAssemblyError as exc:
        if exc.partial is None:
            exc.partial = _partial(accumulator)
        raise


def execute(lm, call):
    if result := _cached(lm, call):
        return result
    backend, provider = _engine(lm, call, False)
    results = []
    # Candidates are separate sequential requests. Bounded parallelism is a
    # follow-up: it must keep output order, per-candidate retry accounting and
    # whole-call caching, and it does not remove per-request input-token charges.
    try:
        for _ in range(call.n):
            for attempt in range(lm.num_retries + 1):
                state = _Attempt()
                try:
                    with error_boundary(lm.model, provider=provider, unexpected=True):
                        _attempt(lm, call, backend, provider, state, results)
                    break
                except Exception as exc:
                    if not _retryable(state, exc, attempt, lm.num_retries):
                        raise
                    time.sleep(_delay(exc, attempt))
            # Completion is irreversible. Advisory pricing and storage never
            # run inside the retry region, even if custom code raises here.
            _price_result(lm, state.result, state.response, provider, call.request)
        result = results[0] if len(results) == 1 else combine(results, model_type=lm.model_type)
        _store(lm, call, result)
        return result
    except BaseException as exc:
        _account_failed_call(lm, results, exc)
        raise


async def aexecute(lm, call):
    if call.cache:
        if result := await asyncio.to_thread(_cached, lm, call):
            return result
    backend, provider = _engine(lm, call, True)
    results = []
    try:
        for _ in range(call.n):
            for attempt in range(lm.num_retries + 1):
                state = _Attempt()
                try:
                    with error_boundary(lm.model, provider=provider, unexpected=True):
                        await _aattempt(lm, call, backend, provider, state, results)
                    break
                except Exception as exc:
                    if not _retryable(state, exc, attempt, lm.num_retries):
                        raise
                    await asyncio.sleep(_delay(exc, attempt))
            await asyncio.to_thread(_price_result, lm, state.result, state.response, provider, call.request)
        result = results[0] if len(results) == 1 else combine(results, model_type=lm.model_type)
        if call.cache:
            # Cancellation may leave this worker writing a complete result,
            # never a partial one. Usage still belongs to this public call.
            await asyncio.to_thread(_store, lm, call, result)
        return result
    except BaseException as exc:
        _account_failed_call(lm, results, exc)
        raise


def finalize(lm, call, result):
    if not result.cache_hit and settings.usage_tracker:
        settings.usage_tracker.add_usage(lm.model, result.usage)
    if not settings.disable_history:
        import datetime
        import uuid

        from dspy.clients.lm15_boundary import history_messages

        entry = {"prompt": call.prompt, "messages": history_messages(call.request),
                 "kwargs": {key: val for key, val in call.options.items() if not key.startswith("api_")},
                 "request": call.request,
                 "response": result.responses[0] if len(result.responses) == 1 else result.responses,
                 "outputs": result.outputs, "usage": result.usage, "cost": result.cost,
                 "timestamp": datetime.datetime.now().isoformat(), "uuid": str(uuid.uuid4()),
                 "model": lm.model, "response_model": result.response_model, "model_type": lm.model_type}
        if result.cost_details:
            entry["cost_details"] = result.cost_details
        adaptations = tuple(a for response in result.responses for a in getattr(response, "adaptations", ()))
        if adaptations:
            entry["adaptations"] = adaptations
        lm.update_history(entry)
    if call.convenience:
        return result.outputs
    return result.responses[0]
