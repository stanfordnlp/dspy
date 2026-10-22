"""
lm15.registry — the one table of named providers.

A routable provider string names a :class:`ProviderDefinition`: a wire
dialect (the adapter class that speaks it), an :class:`AccessPolicy`
(credential chain, static headers, endpoint surfaces, default base URL)
and — for the Chat Completions dialect — the compat preset that names the
server's quirks.  The router, the doctor, the vet surface dump (and
through it the contract's support matrix) and the docs tables all read
this table.  Nothing else lists providers.

A third source of entries, with the same shape, is the caller:
``RouterConfig(providers=(ProviderDefinition.chat(...), ...))`` declares a
provider this table does not list (a hosted gateway, a service lm15 has
not receipted yet).  A declared entry names its compat *object* rather
than a preset name, is routable only through routers built with that
config, and answers ``Resolution.declared`` — it carries no live receipt,
and lm15 says so rather than pretend.

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
- ``access.provider`` equals the entry id (hyphenated form; the underscore
  form is an input alias only — spec/vocabularies.md § Open string
  namespaces, 2026-09-08);
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
    AsyncTypeSafeLM,
    AsyncXaiLM,
    ClaudeCodeLM,
    GeminiLM,
    OpenAIChatLM,
    OpenAICodexLM,
    OpenAILM,
    TypeSafeLM,
    XaiLM,
)

__all__ = [
    "Compat",
    "Dialect",
    "ProviderDefinition",
    "PROVIDERS",
    "canonical_provider",
    "lookup",
]

# The wire formats lm15 speaks.  A dialect is a class; a provider is a
# dialect plus an access policy (plus a compat preset for the chat dialect).
Dialect = Literal["openai-responses", "openai-chat", "anthropic", "gemini", "typesafe"]

# A compat value: a preset name for a registry entry (validated against
# the dialect's table below), or the object itself for a declared entry.
Compat = OpenAIResponsesCompat | OpenAIChatCompat | AnthropicCompat

# Per dialect: the compat class a bound entry may carry as an object, and
# the sync/async classes that speak it.  A dialect absent here cannot bind.
_DIALECTS: dict[str, tuple[type, type, type]] = {
    "openai-responses": (OpenAIResponsesCompat, OpenAILM, AsyncOpenAILM),
    "openai-chat": (OpenAIChatCompat, OpenAIChatLM, AsyncOpenAIChatLM),
    "anthropic": (AnthropicCompat, AnthropicLM, AsyncAnthropicLM),
}


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
    ``compat``          compat preset name (bound registry entries), or the
                        compat object itself (a declared entry, whose
                        preset lives in the caller's code, not lm15's
                        tables).
    ``aliases``         extra input spellings of the provider, canonical
                        form (``fireworks-ai`` for litellm's
                        ``fireworks_ai/``); accepted as ``alias:model``
                        and ``alias/model``, never emitted.
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
    compat: str | Compat | None = None
    placeholder_key: str | None = None
    console_url: str | None = None
    note: str = ""
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.id != canonical_provider(self.id):
            raise ValueError(f"provider id must be hyphenated: {self.id!r}")
        if canonical_provider(self.access.provider) != self.id:
            raise ValueError(f"{self.id}: access policy names provider {self.access.provider!r}")
        if self.placeholder_key is not None and self.access.env_keys:
            raise ValueError(f"{self.id}: a keyless local server declares no env_keys")
        if not isinstance(self.aliases, tuple) or not all(isinstance(a, str) and a for a in self.aliases):
            raise TypeError(f"{self.id}: aliases is a tuple of non-empty strings")
        canonical = tuple(canonical_provider(a) for a in self.aliases)
        if canonical != self.aliases:
            raise ValueError(f"{self.id}: aliases are hyphenated: {canonical!r}")
        if len(set(self.aliases)) != len(self.aliases) or self.id in self.aliases:
            raise ValueError(f"{self.id}: aliases repeat a spelling")
        if self.compat is not None and not isinstance(self.compat, str):
            # A compat object: the dialect's own class, and a place to send
            # the request (a bound entry has no preset table row to read a
            # URL from; a hosted entry renders its host template).
            if self.dialect not in _DIALECTS:
                raise ValueError(f"{self.id}: dialect {self.dialect!r} takes no compat")
            compat_cls = _DIALECTS[self.dialect][0]
            if not isinstance(self.compat, compat_cls):
                raise TypeError(
                    f"{self.id}: compat for dialect {self.dialect!r} is a {compat_cls.__name__}, "
                    f"got {type(self.compat).__name__}"
                )
            if not self.hosted and not self.access.base_url:
                raise ValueError(f"{self.id}: a declared provider names its base_url on the access policy")
            if not self.hosted and self.access.credential_policy != "key":
                # The router binds a bound entry as cls(api_key=..., access=...,
                # compat=...); the OAuth policies build self-resolving
                # constructors that take no access policy, so a bound OAuth
                # entry would silently lose its declaration.
                raise ValueError(
                    f"{self.id}: a declared provider is key-based; credential_policy "
                    f"{self.access.credential_policy!r} needs its own adapter class"
                )
            return
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

    @property
    def spellings(self) -> tuple[str, ...]:
        """Every canonical input spelling: the id, then the aliases."""
        return (self.id, *self.aliases)

    # ─── declaring a provider (RouterConfig(providers=...)) ─────────────

    @classmethod
    def chat(
        cls,
        access: AccessPolicy,
        *,
        compat: OpenAIChatCompat | str,
        aliases: tuple[str, ...] = (),
        placeholder_key: str | None = None,
        console_url: str | None = None,
        note: str = "",
    ) -> ProviderDefinition:
        """A provider speaking the OpenAI Chat Completions wire: ``access``
        names it (``AccessPolicy(provider=..., env_keys=..., base_url=...)``)
        and ``compat`` describes the server's spellings
        (:class:`~lm15.compat.OpenAIChatCompat`)."""
        return cls._bound("openai-chat", access, compat=compat, aliases=aliases, placeholder_key=placeholder_key,
                          console_url=console_url, note=note)

    @classmethod
    def responses(
        cls,
        access: AccessPolicy,
        *,
        compat: OpenAIResponsesCompat | str,
        aliases: tuple[str, ...] = (),
        console_url: str | None = None,
        note: str = "",
    ) -> ProviderDefinition:
        """A provider speaking the OpenAI Responses wire."""
        return cls._bound("openai-responses", access, compat=compat, aliases=aliases, console_url=console_url, note=note)

    @classmethod
    def anthropic(
        cls,
        access: AccessPolicy,
        *,
        compat: AnthropicCompat | str,
        aliases: tuple[str, ...] = (),
        console_url: str | None = None,
        note: str = "",
    ) -> ProviderDefinition:
        """A provider speaking the Anthropic Messages wire."""
        return cls._bound("anthropic", access, compat=compat, aliases=aliases, console_url=console_url, note=note)

    @classmethod
    def _bound(cls, dialect: Dialect, access: AccessPolicy, *, compat, aliases=(), placeholder_key=None,
               console_url=None, note="") -> ProviderDefinition:
        _, adapter, async_adapter = _DIALECTS[dialect]
        return cls(
            id=canonical_provider(access.provider),
            dialect=dialect,
            adapter=adapter,
            async_adapter=async_adapter,
            access=access,
            compat=compat,
            placeholder_key=placeholder_key,
            console_url=console_url,
            note=note,
            aliases=tuple(aliases),
        )


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
    return ProviderDefinition.responses(access, compat=compat, console_url=console_url, note=note)


def _chat_bound(
    access: AccessPolicy,
    *,
    compat: str | None = None,
    placeholder_key: str | None = None,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    return ProviderDefinition.chat(access, compat=compat or access.provider, placeholder_key=placeholder_key,
                                   console_url=console_url, note=note)


def _anthropic_bound(
    access: AccessPolicy,
    *,
    compat: str,
    console_url: str | None = None,
    note: str = "",
) -> ProviderDefinition:
    return ProviderDefinition.anthropic(access, compat=compat, console_url=console_url, note=note)


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
        "typesafe", "typesafe", TypeSafeLM, AsyncTypeSafeLM,
        console_url="https://console.typesafe.ai/keys",
        note="TypeSafe System One (Jev): judgments over declared keys with probabilities; no text generation",
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
    _chat_bound(
        _access.DEEPINFRA,
        console_url="https://deepinfra.com/dash/api_keys",
        note="DeepInfra open-model inference (Chat Completions dialect; models are vendor/name ids)",
    ),
    _chat_bound(
        _access.TOGETHER,
        console_url="https://api.together.ai/settings/projects/~current/api-keys",
        note="Together AI open-model inference (Chat Completions dialect; gpt-oss refuses a forced tool choice client-side — "
             "Together answers it with HTTP 500)",
    ),
    _chat_bound(
        _access.FIREWORKS,
        console_url="https://app.fireworks.ai/settings/users/api-keys",
        note="Fireworks AI open-model inference (Chat Completions dialect; models are accounts/fireworks/models/<name> ids)",
    ),
    _chat_bound(
        _access.PARASAIL,
        console_url="https://www.saas.parasail.io/keys",
        note="Parasail open-model inference (Chat Completions dialect; serverless models)",
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
        note="Azure OpenAI v1 Responses wire ({resource}.openai.azure.com, or AZURE_OPENAI_ENDPOINT = the Foundry root the console shows; model = deployment name; api-key or Entra token)",
    ),
    _hosted(
        _access.AZURE_CHAT, "openai-chat", OpenAIChatLM, AsyncOpenAIChatLM, compat="openai",
        console_url="https://portal.azure.com/",
        note="Azure OpenAI v1 Chat Completions wire (same resource or AZURE_OPENAI_ENDPOINT; also Foundry-sold models such as DeepSeek, Kimi and Grok)",
    ),
    _hosted(
        _access.AZURE_ANTHROPIC, "anthropic", AnthropicLM, AsyncAnthropicLM,
        console_url="https://ai.azure.com/",
        note="Claude in Microsoft Foundry ({resource}.services.ai.azure.com/anthropic or ANTHROPIC_FOUNDRY_BASE_URL; x-api-key or Entra token)",
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
