"""Temporary warning policy for the DSPy 3.4 to 3.5 LM-interface migration.

Warnings belong to public API use, not provider-wire conversion. Remove this
module and the adapter-origin marker with the legacy interfaces in 3.5.
"""

import inspect
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

_GUIDE = "https://dspy.ai/community/normalized-lm-api-migration/"
_adapter_call = ContextVar("dspy_legacy_adapter_message_call", default=None)


def _warn(message: str) -> None:
    # Python 3.10/3.11 lack warnings.warn(skip_file_prefixes=...). Skip DSPy's
    # own forwarding frames for attribution only, never to decide eligibility.
    # Keep Python's filtering rather than storing warning state on LM objects.
    frame = inspect.currentframe()
    stacklevel = 1
    try:
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if module != "dspy" and not module.startswith("dspy."):
                break
            stacklevel += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def warn_legacy_lm() -> None:
    _warn(
        "Implementing custom LMs through BaseLM.forward() or aforward() is deprecated. "
        "Implement an engine with complete(Request) -> Response and pass it to dspy.LM(engine=...) instead. "
        "The old subclass interface remains supported throughout DSPy 3.4 and is scheduled for removal in 3.5, "
        "along with LegacyEngine and AsyncLegacyEngine. "
        f"See {_GUIDE}#custom-engines-and-legacy-plugins."
    )


def warn_legacy_engine() -> None:
    _warn(
        "LegacyEngine and AsyncLegacyEngine are deprecated and scheduled for removal in DSPy 3.5. "
        "They are transition wrappers for 3.4 only. Migrate the underlying implementation to "
        "complete(Request) -> Response, with an async counterpart when needed. "
        f"See {_GUIDE}#custom-engines-and-legacy-plugins."
    )


def warn_legacy_shortcut() -> None:
    _warn(
        "The custom-engine complete_legacy() shortcut is deprecated and scheduled for removal in DSPy 3.5. "
        "Implement complete(Request) -> Response (async complete for async engines). "
        f"See {_GUIDE}#custom-engines-and-legacy-plugins."
    )


def warn_openai_messages(lm, messages) -> None:
    if messages is None:
        return
    origin = _adapter_call.get()
    if origin is not None and origin[0] is lm and origin[1] is messages:
        return
    _warn(
        "Passing OpenAI-style message dictionaries or other message objects to an LM through messages= is deprecated and "
        "scheduled for removal in DSPy 3.5. Use lm(dspy.lm15.Request(...)) with lm15.Message objects; "
        "it returns an lm15.Response. lm('hello') remains supported as a convenience returning a list. "
        f"See {_GUIDE}#migrating-openai-style-messages."
    )


@contextmanager
def adapter_message_call(lm, messages):
    """Identify only the exact dictionary call owned by a 3.4 adapter.

    Normal program users cannot change this internal boundary. Calls with a
    different LM or message object still warn, even when nested or concurrent.
    This is not a general suppression of warnings in the adapter's context.
    """
    token = _adapter_call.set((lm, messages))
    try:
        yield
    finally:
        _adapter_call.reset(token)
