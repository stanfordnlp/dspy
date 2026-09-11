"""
lm15.providers.openai_codex — the OpenAI Responses dialect on a Codex login.

``OpenAICodexLM`` is a *name*, not a behaviour: it is ``OpenAILM`` bound to
``lm15.access.OPENAI_CODEX``. The ChatGPT Codex backend's differences —
bearer OAuth with a ``chatgpt-account-id``, the ``OpenAI-Beta`` and
``originator`` headers, the instructions prefix, ``store: false``,
streaming-only, the ``{"detail": ...}`` error envelope, the ``/models``
shape, no files/batch/images/speech/live — are fields of that policy or
branches on its ``backend`` inside the dialect, at stated points. A port
needs the policy table and the dialect, not this class.
"""

from __future__ import annotations

import os
from typing import ClassVar

from ..access import (  # noqa: F401
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_CODEX_CLIENT_VERSION,
    DEFAULT_CODEX_INSTRUCTIONS,
    DEFAULT_CODEX_ORIGINATOR,
    OPENAI_CODEX,
)
from ..features import ProviderManifest
from .base import Credential, SyncTransport, default_transport
from .openai import MODEL_LIST_HINT, OpenAILM  # noqa: F401


class OpenAICodexLM(OpenAILM):
    """OpenAI Responses adapter authenticated with local Codex CLI OAuth."""

    manifest: ClassVar[ProviderManifest] = OPENAI_CODEX

    def __init__(
        self,
        api_key: Credential | None = None,
        *,
        account_id: str | None = None,
        auth_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
        client_version: str = DEFAULT_CODEX_CLIENT_VERSION,
    ) -> None:
        self.originator = originator
        self.client_version = client_version
        policy = OPENAI_CODEX
        if originator != DEFAULT_CODEX_ORIGINATOR:
            policy = policy.with_headers({"originator": originator})
        if client_version != DEFAULT_CODEX_CLIENT_VERSION:
            from dataclasses import replace

            policy = replace(policy, backend_options={**policy.backend_options, "client_version": client_version})
        super().__init__(
            api_key=api_key,
            transport=transport or default_transport(),
            base_url=base_url,
            profile=None,
            access=policy,
            credentials_path=auth_path,
            account_id=account_id,
        )

    @classmethod
    def from_codex_cli(
        cls,
        *,
        auth_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
    ) -> "OpenAICodexLM":
        return cls(auth_path=auth_path, transport=transport, base_url=base_url, originator=originator)
