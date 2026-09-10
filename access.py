"""
lm15.access — how a dialect adapter reaches a backend, as a value.

An lm15 adapter is three composed things:

- the **dialect**: the class that speaks a wire format (Anthropic Messages,
  OpenAI Responses, OpenAI Chat Completions, Gemini);
- for the chat dialect, a **compat** value describing a server's quirks
  (``OpenAIChatCompat``);
- an **access policy**: this module. Which credential, which headers,
  which endpoint surfaces, which backend variant, which login hint.

Before 2026-09-02 the third thing was a subclass: ``ClaudeCodeLM`` extended
``AnthropicLM``, ``OpenAICodexLM`` extended ``OpenAILM``, ``XaiLM`` extended
``OpenAIChatLM``, each overriding headers, payload defaults, error hints,
and endpoint blocks. Go and Rust have no inheritance, so every port would
have invented its own shape for the same facts. An ``AccessPolicy`` is a
frozen value; a port copies the table below as data and consults it at the
same named points the reference does (spec/auth.md AUTH-9).

What stays behaviour, deliberately:

- ``backend`` names a variant the dialect adapter switches on at a small,
  documented set of points (the ChatGPT Codex backend answers a different
  error envelope, a different models endpoint, and is streaming-first).
  That is a ``match`` in every language, keyed on data.
- Loading a stored subscription credential is per-language code, keyed by
  ``provider`` (``_CREDENTIAL_LOADERS``); the policy says *that* a stored
  credential is used, the loader says *how*.

``ProviderManifest`` is the same class under its earlier name: an adapter's
manifest is its access policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .auth import (
    CLAUDE_CODE_LOGIN_HINT,
    OPENAI_CODEX_LOGIN_HINT,
    XAI_LOGIN_HINT,
    extract_chatgpt_account_id,
    get_claude_code_access_token,
    get_codex_cli_access_token,
    get_xai_access_token,
    usable_xai_credential,
)
from .compat import ANTHROPIC_PRESET_BASE_URLS, OPENAI_CHAT_PRESET_BASE_URLS, OPENAI_RESPONSES_PRESET_BASE_URLS
from .errors import NotConfiguredError
from .credentials import AwsCredentials, BearerToken, CredentialLike, CredentialValue, coerce_credential
from .features import (  # noqa: F401 — re-exported
    AccessPolicy,
    AuthHeader,
    AuthScheme,
    CredentialPolicy,
    EndpointSupport,
    HostSetting,
    HostSpec,
)

# What ``api_key=`` accepts everywhere: a string, an AUTH-2 credential value,
# or a zero-arg provider returning either.  ``Credential`` is the historical
# name for the same thing.
Credential = CredentialLike

# ─── The table ───────────────────────────────────────────────────────

ANTHROPIC_API = AccessPolicy(
    provider="anthropic",
    supports=EndpointSupport(complete=True, stream=True, files=True, batches=True, models=True),
    auth_modes=("x-api-key",),
    env_keys=("ANTHROPIC_API_KEY",),
    auth_scheme=("x-api-key",),
)

DEFAULT_CLAUDE_CODE_VERSION = "2.1.170"
DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."

# models=True: the Anthropic /v1/models endpoint answers to the OAuth
# headers (validated live 2026-08-31, HTTP 200). Files and batch are
# API-key surfaces the subscription token does not carry.
CLAUDE_CODE = AccessPolicy(
    provider="claude-code",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    credential_policy="oauth",
    auth_modes=("claude-code-oauth", "bearer-oauth"),
    auth_scheme=("bearer",),
    headers=(
        ("anthropic-dangerous-direct-browser-access", "true"),
        ("anthropic-beta", "claude-code-20250219,oauth-2025-04-20"),
        ("x-app", "cli"),
        ("user-agent", f"claude-cli/{DEFAULT_CLAUDE_CODE_VERSION}"),
    ),
    login_hint=CLAUDE_CODE_LOGIN_HINT,
    backend="claude-code",
    system_prefix=DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT,
)

OPENAI_API = AccessPolicy(
    provider="openai",
    supports=EndpointSupport(
        complete=True, stream=True, live=True, files=True, batches=True,
        images=True, speech=True, video=True, responses_api=True, models=True,
    ),
    auth_modes=("bearer",),
    env_keys=("OPENAI_API_KEY",),
    enterprise_variants=("azure-openai",),
)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "lm15"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."
# The backend's /models endpoint requires a client_version query parameter
# (a Codex CLI release); any recent release is accepted.
DEFAULT_CODEX_CLIENT_VERSION = "0.147.0"

OPENAI_CODEX = AccessPolicy(
    provider="openai-codex",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    credential_policy="oauth",
    auth_modes=("chatgpt-oauth", "bearer-oauth"),
    headers=(
        ("OpenAI-Beta", "responses=experimental"),
        ("originator", DEFAULT_CODEX_ORIGINATOR),
    ),
    login_hint=OPENAI_CODEX_LOGIN_HINT,
    backend="chatgpt-codex",
    backend_options={"client_version": DEFAULT_CODEX_CLIENT_VERSION},
    system_prefix=DEFAULT_CODEX_INSTRUCTIONS,
    base_url=DEFAULT_CODEX_BASE_URL,
)

OPENAI_CHAT_API = AccessPolicy(
    provider="openai-chat",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("OPENAI_API_KEY",),
)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"

XAI = AccessPolicy(
    provider="xai",
    supports=EndpointSupport(complete=True, stream=True, models=True, images=True, video=True),
    credential_policy="oauth-unless-explicit",
    auth_modes=("bearer", "xai-oauth"),
    env_keys=("XAI_API_KEY",),
    login_hint=XAI_LOGIN_HINT,
    base_url=DEFAULT_XAI_BASE_URL,
)

GEMINI_API = AccessPolicy(
    provider="gemini",
    supports=EndpointSupport(
        complete=True, stream=True, live=True, files=True, batches=True,
        images=True, speech=True, video=True, models=True, caches=True,
    ),
    auth_modes=("query-api-key", "x-goog-api-key"),
    env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    auth_scheme=("x-api-key",),  # the dialect renders it as x-goog-api-key
)

# ─── Responses servers reached through OpenAILM ────────────────────────

# Meta Model API (dev.meta.ai, scraped 2026-09-03 — lm15-contract/scrapes/
# meta/pages).  One bearer key (`LLM|<team id>|<secret>`, authentication.md)
# opens three chat wires and the account surfaces.  `meta` names the
# Responses wire — Meta's recommended default and the only one that carries
# reasoning across turns (protocols.md) — plus the OpenAI-shaped Files
# (files--upload.md: purpose user_data|batch), Images (images--create.md,
# images--edit.md; model muse-image-1.0) and Models (models--list.md)
# endpoints on the same root.  No batch endpoint, no speech synthesis, no
# video generation, no explicit cache resource, and the realtime WebSocket
# is transcription-only (speech-to-text is a surface lm15 does not have).
#
# Env key: META_API_KEY only.  Meta's own docs name the variable
# MODEL_API_KEY (authentication.md); lm15 does not read it: a vendor-less
# name set for some other tool would be picked up and sent to Meta without
# the user asking for it.  The registry note names the documented variable
# so a quickstart user knows what to rename.
META_ENV_KEYS = ("META_API_KEY",)

META = AccessPolicy(
    provider="meta",
    supports=EndpointSupport(complete=True, stream=True, files=True, images=True, responses_api=True, models=True),
    auth_modes=("bearer",),
    env_keys=META_ENV_KEYS,
    base_url=OPENAI_RESPONSES_PRESET_BASE_URLS["meta"],
)

# ─── Chat Completions servers reached through OpenAIChatLM ───────────
#
# These policies are bound onto the chat dialect by the provider registry
# (lm15.registry): the class is OpenAIChatLM, the compat preset of the
# same name names the server's quirks, and this value names the credential
# and the surfaces.  base_url points at the compat table so each URL has
# one copy.

GROQ = AccessPolicy(
    provider="groq",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("GROQ_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["groq"],
)

OPENROUTER = AccessPolicy(
    provider="openrouter",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("OPENROUTER_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["openrouter"],
)

# DeepSeek (api-docs.deepseek.com, scraped 2026-09-03): bearer key from
# platform.deepseek.com, prepaid balance (a drained balance is HTTP 402,
# never a surprise bill).  The same key also opens an Anthropic-format
# endpoint (/anthropic) and a Responses-format endpoint; `deepseek` names
# the Chat Completions wire only — a provider string never guesses a
# protocol.
DEEPSEEK = AccessPolicy(
    provider="deepseek",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("DEEPSEEK_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["deepseek"],
)

# Z.AI (docs.z.ai, scraped 2026-09-03): bearer key from z.ai/manage-apikey,
# general endpoint only.  The GLM Coding Plan is a subscription with its own
# endpoint and its own terms (docs.z.ai/devpack); lm15 does not name it.
ZAI = AccessPolicy(
    provider="zai",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("ZAI_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["zai"],
)

# Moonshot AI / Kimi API Platform (platform.kimi.ai, scraped 2026-09-03):
# bearer key from platform.kimi.ai/console/api-keys, prepaid balance.
# `moonshotai` names the Chat Completions wire (api--overview.md: the
# documented primary; every model, every guide).  The same key also opens a
# Responses wire (kimi-k3 only, `tool_choice: auto` only) and an Anthropic
# Messages wire at /anthropic; neither is registered yet — a provider
# string never guesses a protocol.
#
# Env keys, in order: MOONSHOTAI_API_KEY (lm15's `<provider>_API_KEY`
# convention, the name the maintainer set) then MOONSHOT_API_KEY (the name
# Moonshot's own docs and SDK examples use, api--overview.md).  Both are
# vendor-named, so reading both cannot pick up another tool's secret (the
# reason Meta's vendor-less MODEL_API_KEY is not read).  Stated trade-off:
# with both set to different keys, the lm15-named one wins silently; the
# doctor (AUTH-7) shows the shadowed rung.
MOONSHOTAI_ENV_KEYS = ("MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY")

MOONSHOTAI = AccessPolicy(
    provider="moonshotai",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=MOONSHOTAI_ENV_KEYS,
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["moonshotai"],
)

# Moonshot's Responses wire (responses--create.md): the same key and root,
# kimi-k3 only, stateless (no store, no previous_response_id); reasoning
# returns as summary text and is replayed as-is.  `models` is the same
# GET /models as the chat wire.
MOONSHOTAI_RESPONSES = AccessPolicy(
    provider="moonshotai-responses",
    supports=EndpointSupport(complete=True, stream=True, responses_api=True, models=True),
    auth_modes=("bearer",),
    env_keys=MOONSHOTAI_ENV_KEYS,
    base_url=OPENAI_RESPONSES_PRESET_BASE_URLS["moonshotai"],
)

# Meta's Chat Completions wire (protocols--chat-completions.md): the same
# key, the drop-in for existing messages-array code.  Reasoning is not
# carried across turns here and search grounding is Responses-only; the
# account surfaces (files, images) are reached through `meta`.
META_CHAT = AccessPolicy(
    provider="meta-chat",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=META_ENV_KEYS,
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["meta"],
)

# ─── Anthropic Messages servers reached through AnthropicLM ─────────────

# DeepSeek's Anthropic-format endpoint (guide--anthropic-api.md; live
# 2026-09-03): the same DEEPSEEK_API_KEY, sent as x-api-key.  No /models on
# this root (404) — list models through `deepseek`.  A separate provider
# string because a provider name describes wire behavior: `deepseek` is the
# Chat Completions wire, `deepseek-anthropic` the Messages wire.
DEEPSEEK_ANTHROPIC = AccessPolicy(
    provider="deepseek-anthropic",
    supports=EndpointSupport(complete=True, stream=True),
    auth_modes=("x-api-key",),
    env_keys=("DEEPSEEK_API_KEY",),
    auth_scheme=("x-api-key",),
    base_url=ANTHROPIC_PRESET_BASE_URLS["deepseek"],
)

# Meta's Anthropic-format wire (protocols--messages.md): the same key sent
# as a bearer token (the docs' SDK examples pass it as auth_token, not
# x-api-key).  Stateless; thinking replays through redacted_thinking.  The
# /models root answers on the same base URL with the OpenAI list shape,
# which the dialect's parser reads (live 2026-09-03, 7 entries).
META_ANTHROPIC = AccessPolicy(
    provider="meta-anthropic",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=META_ENV_KEYS,
    auth_scheme=("bearer",),
    base_url=ANTHROPIC_PRESET_BASE_URLS["meta"],
)

# Moonshot's Anthropic-format wire (messages--create.md, securitySchemes
# bearerAuth; the Claude Code guide sets ANTHROPIC_AUTH_TOKEN): the same key
# as a bearer token, kimi-k3 only.  Whether /models answers on the
# /anthropic root is a live finding recorded in the dossier.
MOONSHOTAI_ANTHROPIC = AccessPolicy(
    provider="moonshotai-anthropic",
    supports=EndpointSupport(complete=True, stream=True),
    auth_modes=("bearer",),
    env_keys=MOONSHOTAI_ENV_KEYS,
    auth_scheme=("bearer",),
    base_url=ANTHROPIC_PRESET_BASE_URLS["moonshotai"],
)

# ─── Cloud hosts (spec/auth.md AUTH-10 host policies; changes/2026-09-03-cloud-hosts.md) ───
#
# Every row below is declared from provider documentation and the cloud
# SDKs' resolver sources (lm15-contract/research/cloud-hosts/10-facts-*.md,
# file:line cited there) and awaits its live receipt.  A host is the third
# use of the ``backend`` seam: the dialect is unchanged; the door changes
# the URL, the signing, a closed set of rewrites, and the stream framing.

_AWS_REGION = HostSetting("region", env=("AWS_REGION", "AWS_DEFAULT_REGION"))  # no default: a wrong region is a residency bug
_AWS_WORKSPACE = HostSetting("workspace", env=("ANTHROPIC_AWS_WORKSPACE_ID",))
_GCP_PROJECT = HostSetting("project", env=("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"))
_GCP_LOCATION = HostSetting("location", env=("GOOGLE_CLOUD_LOCATION",), default="global")  # stated trade-off: availability first; the doctor prints it
_AZURE_OPENAI_RESOURCE = HostSetting("resource", env=("AZURE_OPENAI_RESOURCE",))
_AZURE_FOUNDRY_RESOURCE = HostSetting("resource", env=("ANTHROPIC_FOUNDRY_RESOURCE",))
_AZURE_AUTHORITY = HostSetting("authority_host", env=("AZURE_AUTHORITY_HOST",), default="https://login.microsoftonline.com")
_AZURE_SCOPE = HostSetting("scope", env=(), default="https://ai.azure.com/.default")

# Claude Platform on AWS: Anthropic-operated, the Claude API verbatim,
# SigV4 service aws-external-anthropic or an AWS-console API key as
# x-api-key; anthropic-workspace-id on every request; betas pass
# (anthropic-platform-on-aws.md:19-31, :216-218, :244-262, :266-271).
AWS_ANTHROPIC = AccessPolicy(
    provider="aws-anthropic",
    supports=EndpointSupport(complete=True, stream=True),  # batches/models: live cells (the platform says "full Claude API")
    credential_policy="aws-chain",
    auth_modes=("sigv4", "x-api-key"),
    env_keys=("ANTHROPIC_AWS_API_KEY",),
    auth_scheme=("sigv4", "x-api-key"),
    backend="aws-external-anthropic",
    host=HostSpec(
        base_url="https://aws-external-anthropic.{region}.api.aws/v1",
        settings=(_AWS_REGION, _AWS_WORKSPACE),
        required_headers=(("anthropic-workspace-id", "workspace"),),
        sigv4_service="aws-external-anthropic",
    ),
)

# Claude in Amazon Bedrock ("mantle", Opus 4.7 and later): the Messages API
# at /anthropic/v1/messages on AWS-operated infrastructure, plain SSE, SigV4
# service bedrock-mantle or a short-term bearer key as x-api-key; no
# structured outputs, URL/Files sources, server tools, batches, models, or
# anthropic-beta (anthropic-on-bedrock.md:58-70, :138-160, :322, :358-367).
BEDROCK_ANTHROPIC = AccessPolicy(
    provider="bedrock-anthropic",
    supports=EndpointSupport(complete=True, stream=True),
    credential_policy="aws-chain",
    auth_modes=("sigv4", "x-api-key"),
    env_keys=("AWS_BEARER_TOKEN_BEDROCK",),
    auth_scheme=("sigv4", "x-api-key"),
    backend="bedrock-mantle",
    host=HostSpec(
        base_url="https://bedrock-mantle.{region}.api.aws/anthropic/v1",
        settings=(_AWS_REGION,),
        sigv4_service="bedrock-mantle",
    ),
)

# Bedrock's OpenAI Chat Completions door on bedrock-runtime: SigV4 service
# bedrock or AWS_BEARER_TOKEN_BEDROCK (bedrock-openai-chat-completions.md:8,
# :151, :178-228; bedrock-api-keys.md:176-185).  models=False: GET
# /openai/v1/models answers 404 UnknownOperationException under SigV4
# (live 2026-09-03, receipts/2026-09-03-bedrock-chat/probe-models-list.json)
# and the same 404 under a short-term key (live 2026-09-04,
# probe-bearer-models.json), so the docs' listing claim does not hold.
# Live-verified 2026-09-03 (SigV4) and 2026-09-04 (bearer) with
# openai.gpt-oss-20b-1:0.
BEDROCK_CHAT = AccessPolicy(
    provider="bedrock-chat",
    supports=EndpointSupport(complete=True, stream=True),
    credential_policy="aws-chain",
    auth_modes=("sigv4", "bearer"),
    env_keys=("AWS_BEARER_TOKEN_BEDROCK",),
    auth_scheme=("sigv4", "bearer"),
    backend="bedrock-runtime",
    host=HostSpec(
        base_url="https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
        settings=(_AWS_REGION,),
        sigv4_service="bedrock",
    ),
)

# Bedrock-mantle Chat Completions: a different host from BEDROCK_CHAT, not a
# setting.  Live 2026-09-04 (receipts/2026-09-04-bedrock-chat/probe-mantle-chat-*):
# SigV4 service bedrock-mantle; un-versioned model ids (openai.gpt-oss-20b,
# not openai.gpt-oss-20b-1:0); GET /v1/models answers 200 with 55 entries
# under SigV4 and under a short-term key; gpt-oss returns message.reasoning
# (the runtime door inlines <reasoning> in content).  Claude is listed but
# 400 "does not support /v1/chat/completions"; Nova is 404.  AWS's Chat
# Completions page documents listing on this host, not on bedrock-runtime.
BEDROCK_MANTLE_CHAT = AccessPolicy(
    provider="bedrock-mantle-chat",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    credential_policy="aws-chain",
    auth_modes=("sigv4", "bearer"),
    env_keys=("AWS_BEARER_TOKEN_BEDROCK",),
    auth_scheme=("sigv4", "bearer"),
    backend="bedrock-mantle",
    host=HostSpec(
        base_url="https://bedrock-mantle.{region}.api.aws/v1",
        settings=(_AWS_REGION,),
        sigv4_service="bedrock-mantle",
    ),
)

# Azure OpenAI v1: {resource}.openai.azure.com/openai/v1, no api-version,
# api-key header or an Entra bearer token; the model is the DEPLOYMENT name
# (azure-openai-api-lifecycle.md:59-106, :235-250;
# azure-openai-managed-identity.md:213-246).  models=True: the docs call
# listing control-plane only, but GET /openai/v1/models answers 200 with the
# api-key — the resource's whole catalog (dall-e, whisper, gpt-*), not just its
# deployments (receipts/2026-09-04-azure/models.json; cases/azure/models.json).
AZURE = AccessPolicy(
    provider="azure",
    # Live 2026-09-04 on an OpenAI-kind Azure resource: Files and Batches
    # answer on /openai/v1; speech returns raw audio from /audio/speech.
    # Images wait on deployment quota; video is deliberately out of scope.
    supports=EndpointSupport(complete=True, stream=True, live=True, files=True, batches=True, speech=True,
                             responses_api=True, models=True),
    credential_policy="azure-chain",
    auth_modes=("api-key", "entra-oauth"),
    env_keys=("AZURE_OPENAI_API_KEY",),
    auth_scheme=("api-key", "bearer"),
    backend="azure-openai",
    host=HostSpec(
        base_url="https://{resource}.openai.azure.com/openai/v1",
        settings=(_AZURE_OPENAI_RESOURCE, _AZURE_AUTHORITY, _AZURE_SCOPE),
    ),
)

AZURE_CHAT = AccessPolicy(
    provider="azure-chat",
    supports=EndpointSupport(complete=True, stream=True, models=True),  # same /models as `azure`
    credential_policy="azure-chain",
    auth_modes=("api-key", "entra-oauth"),
    env_keys=("AZURE_OPENAI_API_KEY",),
    auth_scheme=("api-key", "bearer"),
    backend="azure-openai",
    host=HostSpec(
        base_url="https://{resource}.openai.azure.com/openai/v1",
        settings=(_AZURE_OPENAI_RESOURCE, _AZURE_AUTHORITY, _AZURE_SCOPE),
    ),
)

# Claude in Microsoft Foundry: {resource}.services.ai.azure.com/anthropic/v1,
# x-api-key or an Entra bearer token; no batches, models, fallbacks.  The docs
# also say `api-key`, but live on an AIServices resource it is 401 while the
# same key under `x-api-key` reaches deployment lookup (404 DeploymentNotFound;
# 2026-09-04, receipts/2026-09-04-azure-anthropic/).
# Sources: anthropic-on-foundry.md:106, :124-143, :176, :305-318, :637-658.
AZURE_ANTHROPIC = AccessPolicy(
    provider="azure-anthropic",
    supports=EndpointSupport(complete=True, stream=True),
    credential_policy="azure-chain",
    auth_modes=("x-api-key", "entra-oauth"),
    env_keys=("ANTHROPIC_FOUNDRY_API_KEY",),
    auth_scheme=("x-api-key", "bearer"),
    backend="azure-foundry",
    host=HostSpec(
        base_url="https://{resource}.services.ai.azure.com/anthropic/v1",
        settings=(_AZURE_FOUNDRY_RESOURCE, _AZURE_AUTHORITY, _AZURE_SCOPE),
    ),
)

# Gemini on Vertex ("Agent Platform"): project + location in the path,
# OAuth bearer with the cloud-platform scope; {location_host} is derived:
# global → aiplatform.googleapis.com, us|eu → aiplatform.{loc}.rep.googleapis.com,
# else {loc}-aiplatform.googleapis.com (vertex-locations.md:40-63, :91).
_VERTEX_BASE = "https://{location_host}/v1/projects/{project}/locations/{location}"

VERTEX = AccessPolicy(
    provider="vertex",
    supports=EndpointSupport(complete=True, stream=True),  # caches/batches/models: live cells
    credential_policy="gcp-chain",
    auth_modes=("google-oauth",),
    env_keys=(),
    auth_scheme=("bearer",),
    backend="vertex",
    host=HostSpec(base_url=_VERTEX_BASE + "/publishers/google", settings=(_GCP_PROJECT, _GCP_LOCATION)),
)

# Express mode: an API key, no project or location (vertex-express-mode.md:84, :181).
VERTEX_EXPRESS = AccessPolicy(
    provider="vertex-express",
    supports=EndpointSupport(complete=True, stream=True),
    credential_policy="key",
    auth_modes=("query-api-key",),
    env_keys=("GOOGLE_API_KEY",),
    auth_scheme=("query-key",),
    backend="vertex-express",
    host=HostSpec(base_url="https://aiplatform.googleapis.com/v1/publishers/google"),
)

# Claude on Vertex: model in the path (:rawPredict / :streamRawPredict),
# anthropic_version in the body as vertex-2023-10-16; no batches, models,
# Files sources (anthropic-on-vertex.md:9-10, :122-126, :358-365).
VERTEX_ANTHROPIC = AccessPolicy(
    provider="vertex-anthropic",
    supports=EndpointSupport(complete=True, stream=True),
    credential_policy="gcp-chain",
    auth_modes=("google-oauth",),
    env_keys=(),
    auth_scheme=("bearer",),
    backend="vertex",
    host=HostSpec(
        base_url=_VERTEX_BASE,
        settings=(_GCP_PROJECT, _GCP_LOCATION),
        paths={
            "messages": "/publishers/anthropic/models/{model}:rawPredict",
            "messages/stream": "/publishers/anthropic/models/{model}:streamRawPredict",
        },
        model_in="path",
        anthropic_version_in="body:vertex-2023-10-16",
    ),
)

CLOUD_HOST_POLICIES: tuple[AccessPolicy, ...] = (
    AZURE, AZURE_CHAT, AZURE_ANTHROPIC,
    AWS_ANTHROPIC, BEDROCK_ANTHROPIC, BEDROCK_CHAT, BEDROCK_MANTLE_CHAT,
    VERTEX, VERTEX_EXPRESS, VERTEX_ANTHROPIC,
)

# Keyless local servers: no env key; the registry supplies the placeholder
# the server accepts when nothing is configured.
OLLAMA = AccessPolicy(
    provider="ollama",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["ollama"],
)

VLLM = AccessPolicy(
    provider="vllm",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["vllm"],
)

SGLANG = AccessPolicy(
    provider="sglang",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["sglang"],
)


# ─── Credential loading, keyed by provider ───────────────────────────


@dataclass(frozen=True, slots=True)
class LoadedCredential:
    credential: Credential = field(repr=False)
    account_id: str | None = None
    source: str = "explicit"  # "explicit" (caller-supplied) or "stored" (local login)


def _load_claude_code(path: str | os.PathLike[str] | None) -> LoadedCredential:
    # Validate now (typed, re-login-guided errors), then re-resolve per
    # request so a long-lived client always sends a fresh token.
    get_claude_code_access_token(path)
    return LoadedCredential(lambda: get_claude_code_access_token(path), source="stored")


def _load_openai_codex(path: str | os.PathLike[str] | None) -> LoadedCredential:
    initial = get_codex_cli_access_token(path)
    account_id = initial.account_id or extract_chatgpt_account_id(initial.access_token)
    return LoadedCredential(lambda: get_codex_cli_access_token(path).access_token, account_id, source="stored")


def _load_xai(path: str | os.PathLike[str] | None) -> LoadedCredential:
    get_xai_access_token(path)
    return LoadedCredential(lambda: get_xai_access_token(path), source="stored")


_CREDENTIAL_LOADERS: dict[str, Callable[[str | os.PathLike[str] | None], LoadedCredential]] = {
    "claude-code": _load_claude_code,
    "openai-codex": _load_openai_codex,
    "xai": _load_xai,
}

_STORED_PROBES: dict[str, Callable[[], bool]] = {
    "xai": usable_xai_credential,
}


def load_credential(
    policy: AccessPolicy,
    api_key: Credential | None,
    *,
    credentials_path: str | os.PathLike[str] | None = None,
) -> LoadedCredential:
    """Resolve the credential an adapter will send under ``policy``.

    An explicit ``api_key`` always wins (AUTH-1). Otherwise a stored-login
    policy loads through its provider's loader; a ``key`` policy with no
    key is a typed not-configured error naming the env keys.
    """
    if api_key is not None and api_key != "":
        return LoadedCredential(api_key)
    loader = _CREDENTIAL_LOADERS.get(policy.provider) if policy.credential_policy != "key" else None
    if loader is not None:
        return loader(credentials_path)
    raise NotConfiguredError(
        f"{policy.provider}: no credential given"
        + (f"; set {' or '.join(policy.env_keys)} or pass api_key=" if policy.env_keys else "; pass api_key="),
        provider=policy.provider,
        env_keys=policy.env_keys,
        credential_hint=policy.login_hint,
    )


def has_stored_credential(policy: AccessPolicy) -> bool:
    """Offline probe (reads files, never the network) for the router's
    ``oauth-unless-explicit`` chain: is a usable login stored locally?"""
    probe = _STORED_PROBES.get(policy.provider)
    return bool(probe()) if probe is not None else False


# AUTH-2 (ratified 2026-09-06, lm15-contract/changes/2026-09-06-decisions.md D1): the
# schemes each credential kind may travel under.  An ApiKey takes the
# policy's first header-carrying scheme in POLICY order; a BearerToken takes
# `bearer` if listed, else `x-api-key` if listed — the TOKEN's order, not the
# policy's, so an Entra token on azure-anthropic (x-api-key, bearer) still
# goes as bearer while a Bedrock short-term key on bedrock-anthropic
# (sigv4, x-api-key) goes in the key header; AwsCredentials sign (sigv4) only.
_ACCEPTED_SCHEMES: dict[str, tuple[AuthScheme, ...]] = {
    "api_key": ("bearer", "x-api-key", "api-key", "query-key"),
    "bearer_token": ("bearer", "x-api-key"),
    "aws": ("sigv4",),
}


def select_scheme(policy: AccessPolicy, credential: CredentialValue) -> AuthScheme:
    """The scheme this credential kind travels under (AUTH-2, AUTH-10).

    ``ApiKey``: the first scheme in the policy's order that carries a key.
    ``BearerToken``: ``bearer`` if the policy lists it, else ``x-api-key`` if
    the policy lists it.  ``AwsCredentials``: ``sigv4`` only.  Anything else
    raises ``NotConfiguredError`` naming the schemes the kind accepts and the
    schemes the door offers.
    """
    accepted = _ACCEPTED_SCHEMES[credential.kind]
    if isinstance(credential, BearerToken):
        for scheme in accepted:
            if scheme in policy.auth_scheme:
                return scheme
    else:
        for scheme in policy.auth_scheme:
            if scheme in accepted:
                return scheme
    raise NotConfiguredError(
        f"{policy.provider}: a {credential.kind} credential cannot travel under "
        f"{'/'.join(policy.auth_scheme)}; it accepts {'/'.join(accepted)}",
        provider=policy.provider,
        env_keys=policy.env_keys,
    )


def _looks_like_jwt(text: str) -> bool:
    """Three base64url segments whose first decodes to a JSON object with
    ``alg`` — the JWS compact form every Entra/OAuth access token uses."""
    import base64
    import json

    parts = text.split(".")
    if len(parts) != 3 or not all(parts):
        return False
    head = parts[0]
    try:
        decoded = base64.urlsafe_b64decode(head + "=" * (-len(head) % 4))
        header = json.loads(decoded)
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(header, dict) and "alg" in header


def auth_header(
    policy: AccessPolicy,
    credential: str | CredentialValue,
    *,
    api_key_header: str = "x-api-key",
) -> tuple[str, str] | None:
    """The (name, value) header that carries ``credential`` under this
    policy, or ``None`` when the scheme is not a header (``sigv4`` signs the
    finished request; ``query-key`` is a query parameter) — the dialect's
    ``_emit`` handles those."""
    value = coerce_credential(credential)
    scheme = select_scheme(policy, value)
    if scheme in ("api-key", "x-api-key") and "bearer" in policy.auth_scheme and _looks_like_jwt(value.value):
        # A plain string reads as an API key, and on this door the key header
        # comes before bearer.  A JWT is never an API key on these doors: it
        # is an Entra/OAuth access token that a token-provider callable
        # (azure-identity's get_bearer_token_provider returns a str) handed
        # over as a string.  Sent as a key it is a bare 401 ("invalid
        # subscription key", live 2026-09-04); name the fix instead.
        raise NotConfiguredError(
            f"{policy.provider}: the credential is a JWT (a bearer token), but a plain string travels as an"
            f" API key here (`{'x-api-key' if scheme == 'x-api-key' else 'api-key'}` header); wrap it:"
            " lm15.credentials.BearerToken(token)",
            provider=policy.provider,
            env_keys=policy.env_keys,
            credential_hint="api_key=lambda: BearerToken(provider())",
        )
    if scheme == "bearer":
        return ("Authorization", f"Bearer {value.value}")
    if scheme == "x-api-key":
        return (api_key_header, value.value)
    if scheme == "api-key":
        return ("api-key", value.value)
    return None
