"""
lm15.providers.claude_code — the Anthropic dialect on a Claude Code login.

``ClaudeCodeLM`` is a *name*, not a behaviour: it is ``AnthropicLM`` bound
to ``lm15.access.CLAUDE_CODE``. Every wire difference from an API-key
client — bearer auth, the ``anthropic-beta`` and ``x-app``/``user-agent``
headers, the required system-prompt prefix, the re-login hint on auth
errors, no files/batch/live — is a field of that policy, consulted by the
dialect at stated points. A port needs the policy table and the dialect,
not this class.
"""

from __future__ import annotations

import os
from typing import ClassVar, Mapping

from ..access import CLAUDE_CODE, DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT, DEFAULT_CLAUDE_CODE_VERSION  # noqa: F401
from ..features import ProviderManifest
from .anthropic import AnthropicLM
from ..adaptation import AdaptationPolicy
from .base import Credential, SyncTransport, default_transport


class ClaudeCodeLM(AnthropicLM):
    """Anthropic Messages adapter authenticated with local Claude Code OAuth.

    ``settings={"client_version": "2.1.290"}`` (or ``claude_code_version=``,
    the same setting under its older name) changes the Claude Code release
    the door claims; see ``lm15.access.DEFAULT_CLAUDE_CODE_VERSION``.  A
    router also reads ``LM15_CLAUDE_CODE_VERSION``; an adapter built by hand
    reads no environment.
    """

    manifest: ClassVar[ProviderManifest] = CLAUDE_CODE

    def __init__(
        self,
        api_key: Credential | None = None,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2023-06-01",
        claude_code_version: str | None = None,
        settings: "Mapping[str, str] | None" = None,
        adaptations: "AdaptationPolicy" = "note",
    ) -> None:
        settings = merge_client_version(settings, claude_code_version, "claude_code_version")
        super().__init__(
            api_key=api_key,
            transport=transport or default_transport(),
            base_url=base_url,
            api_version=api_version,
            access=CLAUDE_CODE,
            credentials_path=credentials_path,
            settings=settings,
            adaptations=adaptations,
        )
        self.claude_code_version = (self.access or CLAUDE_CODE).backend_options["client_version"]

    @classmethod
    def from_claude_code(
        cls,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        claude_code_version: str | None = None,
    ) -> "ClaudeCodeLM":
        return cls(
            credentials_path=credentials_path,
            transport=transport,
            base_url=base_url,
            claude_code_version=claude_code_version,
        )


def merge_client_version(settings: "Mapping[str, str] | None", version: str | None, keyword: str) -> "dict[str, str] | None":
    """``settings`` with the ``client_version`` a named keyword gave; two
    different answers are a configuration error, not a precedence question."""
    if version is None:
        return None if settings is None else dict(settings)
    merged = dict(settings or {})
    if merged.get("client_version", version) != version:
        raise ValueError(f"{keyword}={version!r} and settings client_version={merged['client_version']!r} disagree; pass one")
    merged["client_version"] = version
    return merged
