"""
lm15.registry — the one table of named providers.

A routable provider string names a :class:`ProviderDefinition`: a wire
dialect (the adapter class that speaks it), an :class:`AccessPolicy`
(credential chain, static headers, endpoint surfaces, default base URL)
and — for the Chat Completions dialect — the compat preset that names the
server's quirks.  The router, the doctor, the vet surface dump (and
through it the contract's support matrix) and the docs tables all read
this table.  Nothing else lists providers.

Two kinds of entries share one shape:

- **adapter-owned** — the dialect class carries its own manifest
  (``OpenAILM``, ``AnthropicLM``, ``GeminiLM``, ``XaiLM``, ``ClaudeCodeLM``,
  ``OpenAICodexLM``).  The entry points at the class; ``access`` IS the
  class manifest.
- **bound** — a dialect class with an access policy bound at construction
  plus a compat preset: ``OpenAIChatLM`` for ``groq``, ``openrouter``,
  ``deepseek``, ``zai``, ``ollama``, ``vllm``, ``sglang``; ``AnthropicLM``
  for ``deepseek-anthropic``.  Pure data: adding one is a declaration in
  this file plus a live receipt in the contract, never a new class.

Rules this table enforces (``tests/test_registry.py``):

- a provider string names ONE wire behavior — the same service reachable
  over two dialects is two entries (spec rule: a provider name describes
  wire behavior, not credential ownership);
- ``access.provider`` equals the entry id (hyphenated form; the chat
  dialect's own manifest keeps its historical ``openai_chat`` spelling);
- a bound entry's ``access.base_url`` equals its dialect's compat table
  URL for the same preset — one copy of each URL;
- an entry with ``placeholder_key`` declares no env keys (keyless local
  servers), and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from . import access as _access
from .compat import (
    ANTHROPIC_PRESET_BASE_URLS,
    OPENAI_CHAT_PRESET_BASE_URLS,
    OPENAI_RESPONSES_PRESET_BASE_URLS,
    AnthropicCompat,
    OpenAIChatCompat,
    OpenAIResponsesCompat,
)
from .features import AccessPolicy, CredentialPolicy, EndpointSupport
from .providers import (
    AnthropicLM,
    AsyncAnthropicLM,
    AsyncClaudeCodeLM,
    AsyncGeminiLM,
    AsyncOpenAIChatLM,
    AsyncOpenAICodexLM,
    AsyncOpenAILM,
    AsyncXaiLM,
    ClaudeCodeLM,
    GeminiLM,
    OpenAIChatLM,
    OpenAICodexLM,
    OpenAILM,
    XaiLM,
)

__all__ = [
    "Dialect",
    "ProviderDefinition",
    "PROVIDERS",
    "canonical_provider",
    "lookup",
]

# The wire formats lm15 speaks.  A dialect is a class; a provider is a
# dialect plus an access policy (plus a compat preset for the chat dialect).
Dialect = Literal["openai-responses", "openai-chat", "anthropic", "gemini"]


# Per dialect: the compat preset constructor and the preset → base URL table
# a bound entry is validated against.  A dialect absent here cannot bind.
_COMPAT_TABLES: dict[str, tuple] = {
    "openai-responses": (OpenAIResponsesCompat.preset, OPENAI_RESPONSES_PRESET_BASE_URLS),
    "openai-chat": (OpenAIChatCompat.preset, OPENAI_CHAT_PRESET_BASE_URLS),
    "anthropic": (AnthropicCompat.preset, ANTHROPIC_PRESET_BASE_URLS),
}


def canonical_provider(name: str) -> str:
    """Provider strings are hyphenated (``openai-chat``); the underscore
    spelling is accepted everywhere as a permanent alias."""
    return name.replace("_", "-")


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """Everything lm15 knows about one named provider, as a value.

    ``id``              canonical provider string (hyphenated).
    ``dialect``         the wire format; names which adapter class speaks it.
    ``adapter``         sync LM class; ``async_adapter`` its async mirror.
    ``access``          the credential chain, headers, surfaces and default
                        base URL (``lm15.access``); for a bound entry the
                        router passes this to the dialect constructor.
    ``compat``          Chat Completions preset name (bound entries only).
    ``placeholder_key`` the key a keyless local server accepts when nothing
                        is configured (AUTH-1 last rung); None otherwise.
    ``console_url``     where a human gets a key (docs, doctor hints).
    ``note``            one-line human rationale, surfaced in docs and
                        ``Resolution.describe``.
    """

    id: str
    dialect: Dialect
    adapter: type
    async_adapter: type
    access: AccessPolicy
    compat: str | None = None
    placeholder_key: str | None = None
    console_url: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.id != canonical_provider(self.id):
            raise ValueError(f"provider id must be hyphenated: {self.id!r}")
        if canonical_provider(self.access.provider) != self.id:
            raise ValueError(f"{self.id}: access policy names provider {self.access.provider!r}")
        if self.placeholder_key is not None and self.access.env_keys:
            raise ValueError(f"{self.id}: a keyless local server declares no env_keys")
        if self.hosted:
            # A cloud door (AUTH-10 host): the base URL is a template rendered
            # over the host settings at construction, so the compat-table URL
            # rule does not apply; a compat preset is optional and, when
            # named, must exist for the dialect.
            if self.compat is not None:
                presets, _ = _COMPAT_TABLES[self.dialect]
                presets(self.compat)
            return
        if self.bound:
            if self.compat is None:
                raise ValueError(f"{self.id}: a bound entry names its compat preset")
            presets, urls = _COMPAT_TABLES[self.dialect]
            presets(self.compat)  # raises for an unknown preset
            expected = urls.get(self.compat)
            if self.access.base_url != expected:
                raise ValueError(
                    f"{self.id}: access.base_url {self.access.base_url!r} != "
                    f"compat table {expected!r} for preset {self.compat!r}"
                )

    @property
    def hosted(self) -> bool:
        """True when the access policy names a cloud host (AUTH-10)."""
        return self.access.host is not None

    @property
    def bound(self) -> bool:
        """True when the router binds ``access`` onto the dialect class at
        construction (the class's own manifest is not this provider)."""
        return self.access is not self.adapter.manifest

    @property
    def env_keys(self) -> tuple[str, ...]:
        return self.access.env_keys

    @property
    def credential_policy(self) -> CredentialPolicy:
        return self.access.credential_policy

    @property
    def supports(self) -> EndpointSupport:
        return self.access.supports

    @property
    def base_url(self) -> str | None:
        return self.access.base_url


def _adapter_owned(
    id: str,
    dialect: Dialect,
    adapter: type,
    async_adapter: type,
    *,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    return ProviderDefinition(
        id=id,
        dialect=dialect,
        adapter=adapter,
        async_adapter=async_adapter,
        access=adapter.manifest,
        console_url=console_url,
        note=note,
    )


def _responses_bound(
    access: AccessPolicy,
    *,
    compat: str,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    return ProviderDefinition(
        id=access.provider,
        dialect="openai-responses",
        adapter=OpenAILM,
        async_adapter=AsyncOpenAILM,
        access=access,
        compat=compat,
        console_url=console_url,
        note=note,
    )


def _chat_bound(
    access: AccessPolicy,
    *,
    compat: str | None = None,
    placeholder_key: str | None = None,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    return ProviderDefinition(
        id=access.provider,
        dialect="openai-chat",
        adapter=OpenAIChatLM,
        async_adapter=AsyncOpenAIChatLM,
        access=access,
        compat=compat or access.provider,
        placeholder_key=placeholder_key,
        console_url=console_url,
        note=note,
    )


def _anthropic_bound(
    access: AccessPolicy,
    *,
    compat: str,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    return ProviderDefinition(
        id=access.provider,
        dialect="anthropic",
        adapter=AnthropicLM,
        async_adapter=AsyncAnthropicLM,
        access=access,
        compat=compat,
        console_url=console_url,
        note=note,
    )


def _hosted(
    access: AccessPolicy,
    dialect: Dialect,
    adapter: type,
    async_adapter: type,
    *,
    compat: str | None = None,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    """A cloud door: an existing dialect behind a host (changes/2026-09-03-cloud-hosts.md).
    Declared from documentation until its live receipt lands."""
    return ProviderDefinition(
        id=access.provider,
        dialect=dialect,
        adapter=adapter,
        async_adapter=async_adapter,
        access=access,
        compat=compat,
        console_url=console_url,
        note=note,
    )


# Declaration order is presentation order (docs tables, `known providers`
# lists are sorted separately).  Adapter-owned entries first, then the
# chat-bound services, then the keyless local servers.
_DEFINITIONS: tuple[ProviderDefinition, ...] = (
    _adapter_owned(
        "openai", "openai-responses", OpenAILM, AsyncOpenAILM,
        console_url="https://platform.openai.com/api-keys",
        note="OpenAI Responses API",
    ),
    _adapter_owned(
        "openai-chat", "openai-chat", OpenAIChatLM, AsyncOpenAIChatLM,
        console_url="https://platform.openai.com/api-keys",
        note="OpenAI Chat Completions dialect (the de-facto standard other servers speak)",
    ),
    _adapter_owned(
        "anthropic", "anthropic", AnthropicLM, AsyncAnthropicLM,
        console_url="https://console.anthropic.com",
        note="Anthropic Messages API",
    ),
    _adapter_owned(
        "gemini", "gemini", GeminiLM, AsyncGeminiLM,
        console_url="https://aistudio.google.com/apikey",
        note="Google Gemini API",
    ),
    _adapter_owned(
        "xai", "openai-chat", XaiLM, AsyncXaiLM,
        console_url="https://console.x.ai",
        note="xAI Grok (Chat Completions dialect; XAI_API_KEY or subscription OAuth)",
    ),
    _adapter_owned(
        "claude-code", "anthropic", ClaudeCodeLM, AsyncClaudeCodeLM,
        note="Claude subscription through the local `claude` CLI login",
    ),
    _adapter_owned(
        "openai-codex", "openai-responses", OpenAICodexLM, AsyncOpenAICodexLM,
        note="ChatGPT subscription through the local `codex` CLI login",
    ),
    _chat_bound(
        _access.GROQ,
        console_url="https://console.groq.com/keys",
        note="Groq Cloud (Chat Completions dialect)",
    ),
    _chat_bound(
        _access.OPENROUTER,
        console_url="https://openrouter.ai/keys",
        note="OpenRouter (Chat Completions dialect)",
    ),
    _chat_bound(
        _access.DEEPSEEK,
        console_url="https://platform.deepseek.com/api_keys",
        note="DeepSeek (Chat Completions dialect; thinking mode on by default)",
    ),
    _anthropic_bound(
        _access.DEEPSEEK_ANTHROPIC,
        compat="deepseek",
        console_url="https://platform.deepseek.com/api_keys",
        note="DeepSeek over the Anthropic Messages wire (same key as `deepseek`; no model listing)",
    ),
    _chat_bound(
        _access.ZAI,
        console_url="https://z.ai/manage-apikey/apikey-list",
        note="Z.AI GLM (Chat Completions dialect; general endpoint, not the Coding Plan)",
    ),
    _chat_bound(
        _access.MOONSHOTAI,
        console_url="https://platform.kimi.ai/console/api-keys",
        note="Moonshot AI Kimi (Chat Completions dialect; kimi-k3 takes reasoning effort low|high|max, kimi-k2.6 takes effort off; "
             "Moonshot's docs call the key MOONSHOT_API_KEY — read after MOONSHOTAI_API_KEY)",
    ),
    _responses_bound(
        _access.MOONSHOTAI_RESPONSES,
        compat="moonshotai",
        console_url="https://platform.kimi.ai/console/api-keys",
        note="Moonshot AI Kimi over the Responses wire (same key as `moonshotai`; kimi-k3 only; stateless — reasoning replays as summary text; web_search built-in)",
    ),
    _anthropic_bound(
        _access.MOONSHOTAI_ANTHROPIC,
        compat="moonshotai",
        console_url="https://platform.kimi.ai/console/api-keys",
        note="Moonshot AI Kimi over the Anthropic Messages wire (same key as `moonshotai`, bearer token; kimi-k3 only)",
    ),
    _responses_bound(
        _access.META,
        compat="meta",
        console_url="https://dev.meta.ai/",
        note="Meta Model API — Muse Spark over the Responses wire (reasoning replay, web_search), plus Files, Images (muse-image-1.0) and Models; Meta's docs call the key MODEL_API_KEY — export it as META_API_KEY",
    ),
    _chat_bound(
        _access.META_CHAT,
        compat="meta",
        console_url="https://dev.meta.ai/",
        note="Meta Model API over the Chat Completions wire (same key as `meta`; no cross-turn reasoning)",
    ),
    _anthropic_bound(
        _access.META_ANTHROPIC,
        compat="meta",
        console_url="https://dev.meta.ai/",
        note="Meta Model API over the Anthropic Messages wire (same key as `meta`; bearer token)",
    ),
    # ─── Cloud hosts (documentation-evidenced; changes/2026-09-03-cloud-hosts.md) ───
    _hosted(
        _access.AZURE, "openai-responses", OpenAILM, AsyncOpenAILM,
        console_url="https://portal.azure.com/",
        note="Azure OpenAI v1 Responses wire ({resource}.openai.azure.com; model = deployment name; api-key or Entra token)",
    ),
    _hosted(
        _access.AZURE_CHAT, "openai-chat", OpenAIChatLM, AsyncOpenAIChatLM, compat="openai",
        console_url="https://portal.azure.com/",
        note="Azure OpenAI v1 Chat Completions wire (same resource; also Foundry-sold models such as DeepSeek and Grok)",
    ),
    _hosted(
        _access.AZURE_ANTHROPIC, "anthropic", AnthropicLM, AsyncAnthropicLM,
        console_url="https://ai.azure.com/",
        note="Claude in Microsoft Foundry ({resource}.services.ai.azure.com/anthropic; api-key, x-api-key or Entra token)",
    ),
    _hosted(
        _access.AWS_ANTHROPIC, "anthropic", AnthropicLM, AsyncAnthropicLM,
        console_url="https://console.aws.amazon.com/",
        note="Claude Platform on AWS (Anthropic-operated; SigV4 or ANTHROPIC_AWS_API_KEY; needs AWS_REGION and ANTHROPIC_AWS_WORKSPACE_ID)",
    ),
    _hosted(
        _access.BEDROCK_ANTHROPIC, "anthropic", AnthropicLM, AsyncAnthropicLM,
        console_url="https://console.aws.amazon.com/bedrock/",
        note="Claude in Amazon Bedrock (bedrock-mantle, Opus 4.7 and later; SigV4 or AWS_BEARER_TOKEN_BEDROCK; needs AWS_REGION)",
    ),
    _hosted(
        _access.BEDROCK_CHAT, "openai-chat", OpenAIChatLM, AsyncOpenAIChatLM, compat="bedrock",
        console_url="https://console.aws.amazon.com/bedrock/",
        note="Amazon Bedrock over the OpenAI Chat Completions wire (bedrock-runtime /openai/v1; SigV4 or AWS_BEARER_TOKEN_BEDROCK)",
    ),
    _hosted(
        _access.BEDROCK_MANTLE_CHAT, "openai-chat", OpenAIChatLM, AsyncOpenAIChatLM, compat="bedrock-mantle",
        console_url="https://console.aws.amazon.com/bedrock/",
        note="Amazon Bedrock Chat Completions on bedrock-mantle (un-versioned ids, GET /v1/models; SigV4 or AWS_BEARER_TOKEN_BEDROCK)",
    ),
    _hosted(
        _access.VERTEX, "gemini", GeminiLM, AsyncGeminiLM,
        console_url="https://console.cloud.google.com/vertex-ai",
        note="Gemini on Google Cloud (Agent Platform); ADC chain; needs GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION defaults to global",
    ),
    _hosted(
        _access.VERTEX_EXPRESS, "gemini", GeminiLM, AsyncGeminiLM,
        console_url="https://console.cloud.google.com/vertex-ai/studio",
        note="Agent Platform express mode: GOOGLE_API_KEY as ?key=, no project or location",
    ),
    _hosted(
        _access.VERTEX_ANTHROPIC, "anthropic", AnthropicLM, AsyncAnthropicLM,
        console_url="https://console.cloud.google.com/vertex-ai/model-garden",
        note="Claude on Google Cloud (rawPredict; model in the path, anthropic_version in the body)",
    ),
    _chat_bound(_access.OLLAMA, placeholder_key="ollama", note="local ollama server (keyless)"),
    _chat_bound(_access.VLLM, placeholder_key="EMPTY", note="local vLLM server (keyless)"),
    _chat_bound(_access.SGLANG, placeholder_key="EMPTY", note="local SGLang server (keyless)"),
)

PROVIDERS: Mapping[str, ProviderDefinition] = MappingProxyType({d.id: d for d in _DEFINITIONS})


def lookup(name: str) -> ProviderDefinition | None:
    """The definition for a provider string in either spelling, or None."""
    return PROVIDERS.get(canonical_provider(name))
