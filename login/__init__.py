"""
lm15.login — managed authentication: sign in once, use everywhere.

spec/auth.md AUTH-12–26 (ratified core, 2026-09-22).  Python is the first
implementation; the contract, not this module, is the authority.

    from lm15.login import Auth, TerminalUI

    auth = Auth.local()                        # reads nothing yet
    auth.login("xai", ui=TerminalUI())         # device code in the terminal; saved on success
    router = LMRouter(RouterConfig(auth=auth)) # explicit keys still win; otherwise the saved login

    from lm15.interactive import connect       # the short path
    with connect() as lm:
        lm.complete(messages=[Message.user("hi")])

What this module does not do: read another tool's login unless you
explicitly configure it as an external source; fall back to an
environment key after a login fails or is signed out; start a login
during an ordinary request; keep any conversation state.

``lm15.auth`` (the legacy CLI-credential helpers and ``login_xai``) keeps
working; ``login_xai`` now runs the same xAI flow as ``Auth.login("xai")``.
"""

from __future__ import annotations

from ..errors import AUTH_OPERATION_REASONS, AuthOperationError
from .engine import ATTEMPT_LIFETIME_S, LoginCancelled
from .bound import BoundClient, model_choices
from .manager import AsyncAuth, Auth, ForgetResult
from .store import FileStore, MemoryStore, Store, default_store_path
from .terminal import TerminalUI
from .types import (
    AuthUI,
    AuthUrlNotice,
    Connection,
    ConnectionStatus,
    DeviceCodeNotice,
    InfoNotice,
    LoginMethod,
    ManualCodePrompt,
    MethodField,
    ModelChoice,
    ModelSelection,
    Notice,
    ProgressNotice,
    Prompt,
    ProviderDescriptor,
    SecretPrompt,
    SelectOption,
    SelectPrompt,
    TextPrompt,
    Verification,
)

__all__ = [
    "ATTEMPT_LIFETIME_S",
    "AUTH_OPERATION_REASONS",
    "AsyncAuth",
    "Auth",
    "AuthOperationError",
    "BoundClient",
    "AuthUI",
    "AuthUrlNotice",
    "Connection",
    "ConnectionStatus",
    "DeviceCodeNotice",
    "FileStore",
    "ForgetResult",
    "InfoNotice",
    "LoginCancelled",
    "LoginMethod",
    "ManualCodePrompt",
    "MemoryStore",
    "MethodField",
    "ModelChoice",
    "ModelSelection",
    "Notice",
    "ProgressNotice",
    "Prompt",
    "ProviderDescriptor",
    "SecretPrompt",
    "SelectOption",
    "SelectPrompt",
    "Store",
    "TerminalUI",
    "TextPrompt",
    "Verification",
    "default_store_path",
    "model_choices",
    "providers",
]


class _Providers:
    """``providers.xai``, ``providers.openai_codex`` … — discoverable names
    for the routes a manager can connect (AUTH-13.1).  Attribute access
    only reads definitions; unknown names raise ``AttributeError``."""

    def __getattr__(self, name: str) -> ProviderDescriptor:
        from .flows import descriptor

        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return descriptor(name.replace("_", "-"))
        except KeyError:
            raise AttributeError(f"no provider {name!r}; see Auth.providers()") from None

    def __dir__(self) -> list[str]:
        from .flows import provider_ids

        return [p.replace("-", "_") for p in provider_ids()]

    def __iter__(self):
        from .flows import descriptor, provider_ids

        return (descriptor(p) for p in provider_ids())


providers = _Providers()
