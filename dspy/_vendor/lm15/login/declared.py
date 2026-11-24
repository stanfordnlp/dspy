"""
lm15.login.declared — routes that exist only for a managed connection.

``kimi-code`` (Kimi Code subscription: Anthropic Messages wire at
``api.kimi.com/coding``) and ``github-copilot`` (Copilot: Chat Completions
wire at the account's Copilot host) have no live wire receipt in
lm15-contract, so they are **not** in the receipted provider registry
(a registry row is a support claim; AUTH-26 forbids one without evidence).
They are *declared providers*, the same mechanism an application uses for
a gateway lm15 has never seen: routed, marked ``Resolution.declared``
("no lm15 receipts"), and added to a router automatically only when that
router carries a managed ``Auth`` — the only way to have a credential
for them.  Nothing here reads a credential.
"""

from __future__ import annotations

from ..compat import AnthropicCompat, OpenAIChatCompat
from ..features import AccessPolicy, EndpointSupport
from ..registry import ProviderDefinition

__all__ = ["DECLARED_PROVIDERS", "KIMI_CODE", "GITHUB_COPILOT", "KIMI_CODE_BASE_URL", "COPILOT_DEFAULT_BASE_URL"]

KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"
COPILOT_DEFAULT_BASE_URL = "https://api.individual.githubcopilot.com"

KIMI_CODE = ProviderDefinition.anthropic(
    AccessPolicy(
        provider="kimi-code",
        supports=EndpointSupport(complete=True, stream=True),
        auth_modes=("bearer",),
        env_keys=(),
        auth_scheme=("bearer",),
        base_url=KIMI_CODE_BASE_URL,
    ),
    compat=AnthropicCompat(),
    note="Kimi Code subscription over the Anthropic Messages wire (managed login only; no lm15 wire receipt yet)",
)

GITHUB_COPILOT = ProviderDefinition.chat(
    AccessPolicy(
        provider="github-copilot",
        supports=EndpointSupport(complete=True, stream=True, models=True),
        auth_modes=("bearer",),
        env_keys=(),
        auth_scheme=("bearer",),
        headers=(
            ("User-Agent", "GitHubCopilotChat/0.35.0"),
            ("Editor-Version", "vscode/1.107.0"),
            ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
            ("Copilot-Integration-Id", "vscode-chat"),
        ),
        base_url=COPILOT_DEFAULT_BASE_URL,
    ),
    compat=OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
    ),
    note="GitHub Copilot over the Chat Completions wire (managed login only; the account's host comes from the token; "
         "no lm15 wire receipt yet)",
)

DECLARED_PROVIDERS: tuple[ProviderDefinition, ...] = (KIMI_CODE, GITHUB_COPILOT)
