"""Adapt a legacy custom LM without invoking BaseLM's public-call bookkeeping."""

from dspy.clients._deprecation import warn_legacy_engine
from dspy.clients.engines.base import validate_request
from dspy.clients.errors import wrap_error
from dspy.clients.lm15_boundary import request_kwargs, response_value
from dspy.lm15 import Request, Response
from dspy.utils.exceptions import LMError, LMUnsupportedFeatureError


class _LegacyConfig:
    """Borrow a plugin; its runtime state and lifecycle remain caller-owned.

    No retries or caching are added here. Third-party forward implementations
    may perform their own; this wrapper cannot disable undocumented behavior.
    """

    def __init__(self, lm, *, _implicit=False):
        from dspy.clients.base_lm import BaseLM
        from dspy.clients.lm import LM

        if not isinstance(lm, BaseLM):
            raise TypeError("LegacyEngine requires a BaseLM instance")
        if type(lm) is LM and not {"forward", "aforward"}.intersection(vars(lm)):
            raise TypeError("Use LiteLLMEngine for the built-in LM, not a nested LegacyEngine")
        if getattr(type(lm), "forward_contract", "legacy") != "legacy":
            raise TypeError("The removed DSPy 3.3 typed_lm contract is not a legacy plugin")
        if not _implicit:
            # Automatic wrapping already warns at the public LM call boundary.
            warn_legacy_engine()
        self.lm = lm
        self._closed = False

    def _arguments(self, request):
        validate_request(request)
        if self._closed:
            raise RuntimeError("Engine is closed")
        if request.model != self.lm.model:
            raise ValueError("Request.model must match the wrapped legacy LM")
        data = request_kwargs(request, "chat")
        messages = data.pop("messages")
        # Explicitly neutralize absent generation defaults; do not mutate the
        # plugin or make a runtime copy that would lose its call-state updates.
        client_keys = {
            "api_key", "api_base", "base_url", "api_version", "headers", "extra_headers",
            "timeout", "azure_ad_token_provider", "organization", "project",
        }
        kwargs = {key: None for key in self.lm.kwargs if key not in client_keys}
        kwargs.update(data)
        kwargs["n"] = 1
        return {"prompt": None, "messages": messages, **kwargs}

    def _response(self, raw, request):
        if isinstance(raw, Response):
            return raw
        return response_value(raw, "chat", request)

    def __repr__(self):
        return f"{type(self).__name__}(plugin={type(self.lm).__name__})"


class LegacyEngine(_LegacyConfig):
    """Deprecated 3.4 transition wrapper, scheduled for removal in DSPy 3.5.

    Migrate the underlying implementation to complete(Request) -> Response.
    Wrapping it explicitly does not extend the legacy interface's lifetime.
    """

    def complete_legacy(self, lm, request, *, prompt=None, messages=None, call_kwargs=None):
        from dspy.clients.call_result import CallResult

        raw = self.lm.forward(prompt=prompt, messages=messages, **(call_kwargs or {}))
        return CallResult.legacy(self.lm, raw, kwargs=call_kwargs)

    def complete(self, request: Request):
        kwargs = self._arguments(request)
        try:
            return self._response(self.lm.forward(**kwargs), request)
        except LMError:
            raise
        except Exception as exc:
            raise wrap_error(exc, model=request.model) from exc

    def stream(self, request: Request):
        validate_request(request)
        raise LMUnsupportedFeatureError(
            "Legacy forward() has no standard streaming contract. "
            "Use the plugin's ordinary streaming path or provide a native streaming engine.",
            model=request.model, features=["stream"],
        )

    def close(self):
        # The plugin may be used elsewhere; never close a borrowed object.
        self._closed = True


class AsyncLegacyEngine(_LegacyConfig):
    """Deprecated async transition wrapper, scheduled for removal in DSPy 3.5."""

    async def complete_legacy(self, lm, request, *, prompt=None, messages=None, call_kwargs=None):
        from dspy.clients.call_result import CallResult

        raw = await self.lm.aforward(prompt=prompt, messages=messages, **(call_kwargs or {}))
        return CallResult.legacy(self.lm, raw, kwargs=call_kwargs)

    async def complete(self, request: Request):
        kwargs = self._arguments(request)
        try:
            return self._response(await self.lm.aforward(**kwargs), request)
        except LMError:
            raise
        except Exception as exc:
            raise wrap_error(exc, model=request.model) from exc

    def stream(self, request: Request):
        validate_request(request)
        raise LMUnsupportedFeatureError(
            "Legacy aforward() has no standard streaming contract.",
            model=request.model, features=["stream"],
        )

    async def aclose(self):
        self._closed = True
