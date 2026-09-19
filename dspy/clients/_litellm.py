from __future__ import annotations

import functools
import sys
import types
from typing import Any

from dspy.utils.lazy_import import require


@functools.cache
def _configure_litellm_defaults(litellm: types.ModuleType) -> None:
    """Apply DSPy's global LiteLLM defaults once when LiteLLM is first imported."""
    litellm.telemetry = False
    litellm.cache = None  # By default we disable LiteLLM cache and use DSPy on-disk cache.
    if not getattr(litellm, "_dspy_logging_configured", False):
        litellm.suppress_debug_info = True
        litellm._dspy_logging_configured = True


def _materialize_litellm(litellm: types.ModuleType) -> None:
    """Force LiteLLM's lazy module to execute, or raise the missing dependency error."""
    # `require()` returns either an importlib LazyLoader-backed module or a _MissingModule.
    # Accessing a real LiteLLM attribute forces LazyLoader execution; on _MissingModule it raises
    # the helpful install-hint ImportError immediately at the DSPy call site.
    _completion = litellm.completion


@functools.cache
def get_litellm(*, feature: str) -> Any:
    """Import LiteLLM, apply DSPy's defaults once, and return the module."""
    litellm = require("litellm", extra="litellm", feature=feature)
    _materialize_litellm(litellm)
    _configure_litellm_defaults(litellm)
    return litellm


def is_litellm_context_window_error(error: Exception) -> bool:
    """Return whether an exception is LiteLLM's context-window error, if LiteLLM is loaded."""
    litellm_module = sys.modules.get("litellm")
    context_window_error = getattr(litellm_module, "ContextWindowExceededError", None)
    return context_window_error is not None and isinstance(error, context_window_error)


def normalize_litellm_error(error: Exception, model: str | None = None) -> Exception:
    """Convert LiteLLM provider exceptions to DSPy exception hierarchy."""
    import dspy.utils.exceptions as exc

    if isinstance(error, exc.DSPyError):
        return error

    litellm_module = sys.modules.get("litellm")
    if litellm_module is not None:
        cw_err = getattr(litellm_module, "ContextWindowExceededError", None)
        rl_err = getattr(litellm_module, "RateLimitError", None)
        auth_err = getattr(litellm_module, "AuthenticationError", None)
        inv_err = getattr(litellm_module, "InvalidRequestError", None)
        bad_req_err = getattr(litellm_module, "BadRequestError", None)
        not_found_err = getattr(litellm_module, "NotFoundError", None)
        unsupported_err = getattr(litellm_module, "UnsupportedParamsError", None)
        conn_err = getattr(litellm_module, "APIConnectionError", None)
        server_err = getattr(litellm_module, "ServiceUnavailableError", None)
        internal_err = getattr(litellm_module, "InternalServerError", None)
        api_err = getattr(litellm_module, "APIError", None)
        timeout_err = getattr(litellm_module, "Timeout", None)

        if cw_err and isinstance(error, cw_err):
            return exc.ContextWindowExceededError(str(error), model=model)
        if rl_err and isinstance(error, rl_err):
            return exc.LMRateLimitError(str(error), model=model)
        if auth_err and isinstance(error, auth_err):
            return exc.LMAuthError(str(error), model=model)
        if unsupported_err and isinstance(error, unsupported_err):
            return exc.LMUnsupportedFeatureError(str(error), model=model)
        if not_found_err and isinstance(error, not_found_err):
            return exc.LMUnsupportedModelError(str(error), model=model)
        if (inv_err and isinstance(error, inv_err)) or (bad_req_err and isinstance(error, bad_req_err)):
            return exc.LMInvalidRequestError(str(error), model=model)
        if conn_err and isinstance(error, conn_err):
            return exc.LMTransportError(str(error), model=model)
        if (server_err and isinstance(error, server_err)) or (internal_err and isinstance(error, internal_err)):
            return exc.LMServerError(str(error), model=model)
        if timeout_err and isinstance(error, timeout_err):
            return exc.LMTimeoutError(str(error), model=model)
        if api_err and isinstance(error, api_err):
            return exc.LMProviderError(str(error), model=model)

    return exc.LMError(str(error), model=model)
