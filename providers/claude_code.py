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
from typing import ClassVar

from ..access import CLAUDE_CODE, DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT, DEFAULT_CLAUDE_CODE_VERSION  # noqa: F401
from ..features import ProviderManifest
from .anthropic import AnthropicLM
from .base import Credential, SyncTransport, default_transport


class ClaudeCodeLM(AnthropicLM):
    """Anthropic Messages adapter authenticated with local Claude Code OAuth."""

    manifest: ClassVar[ProviderManifest] = CLAUDE_CODE

    def __init__(
        self,
        api_key: Credential | None = None,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2023-06-01",
        claude_code_version: str = DEFAULT_CLAUDE_CODE_VERSION,
    ) -> None:
        self.claude_code_version = claude_code_version
        policy = CLAUDE_CODE
        if claude_code_version != DEFAULT_CLAUDE_CODE_VERSION:
            policy = policy.with_headers({"user-agent": f"claude-cli/{claude_code_version}"})
        super().__init__(
            api_key=api_key,
            transport=transport or default_transport(),
            base_url=base_url,
            api_version=api_version,
            access=policy,
            credentials_path=credentials_path,
        )

    @classmethod
    def from_claude_code(
        cls,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        claude_code_version: str = DEFAULT_CLAUDE_CODE_VERSION,
    ) -> "ClaudeCodeLM":
        return cls(
            credentials_path=credentials_path,
            transport=transport,
            base_url=base_url,
            claude_code_version=claude_code_version,
        )
