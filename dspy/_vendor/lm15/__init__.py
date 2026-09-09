"""lm15 — a provider-neutral, low-level foundation for talking to LLM APIs.

lm15 is deliberately NOT a user-facing convenience layer. It is the dependency
for libraries that want to build their own take on the right Python API for
AI systems: one canonical representation (Request/Response/Message/Part,
stream events, errors), exact serde for it, and adapters that translate it to
and from each provider's wire format — stdlib-only, with its own HTTP
transport. Build your DSL on top; let lm15 handle the providers.

Conformance to the canonical representation is pinned by the lm15-contract
corpus; this package is the reference implementation, not the spec.

Quick tour:

    from lm15 import AnthropicLM, Request, Message

    lm = AnthropicLM(api_key="...")
    response = lm.complete(Request(model="claude-sonnet-4-5",
                                   messages=(Message.user("hello"),)))
    response.text          # canonical accessors
    for event in lm.stream(request): ...   # canonical stream events

The top level is curated for application developers.  Deeper audiences
import one module (API review, 2026-07-13):
`lm15.serde` (the complete to_dict/from_dict pairs), `lm15.providers`
(BaseProviderLM, ProviderDialect, transports protocols, Credential),
`lm15.errors` (map_http_error and friends), `lm15.registry` (the one
table of named providers: PROVIDERS / ProviderDefinition — dialect,
access policy, compat preset, console URL), `lm15.router` (the
RouteRule/DEFAULT_RULES tables and the ADAPTERS/CHAT_PRESET_ROUTES views
of the registry),
`lm15.profiles` / `lm15.compat` (profile and compat-policy machinery),
`lm15.sse`, `lm15.transports` (HTTP plumbing), `lm15.live` (realtime
sessions; optional `websockets` dependency), `lm15.auth` (local
subscription credentials: locked, atomic, double-checked refresh),
`lm15.authkit` (login-flow primitives: PKCE, device-code polling,
loopback listener, credential store), `lm15.doctor` (`explain_auth`:
rung-by-rung credential resolution, no secrets rendered), and
`lm15.vet` (the conformance shim CLI: `python -m lm15.vet`).

The lowercase part-factory helpers (`text`, `image`, `tool_call`, ...) live in
`lm15.types`, not at the top level — generic lowercase names at package top
level invite collisions with user code.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("lm15")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

# ── Canonical types ──────────────────────────────────────────────────
from .batch import AsyncBatchJob, BatchJob
from .video_jobs import AsyncVideoJob, VideoJob
from .live import Turn
from .types import (
    # request/response core
    Request,
    Response,
    Message,
    Usage,
    Config,
    CacheConfig,
    Reasoning,
    ToolChoice,
    ErrorDetail,
    ContinuationState,
    # parts
    TextPart,
    ThinkingPart,
    RefusalPart,
    CitationPart,
    ImagePart,
    AudioPart,
    VideoPart,
    DocumentPart,
    BinaryPart,
    ToolCallPart,
    ToolResultPart,
    # tools
    FunctionTool,
    BuiltinTool,
    ToolCallInfo,
    # logprobs
    TokenLogprob,
    TopLogprob,
    # streaming
    StreamStartEvent,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    TextDelta,
    ThinkingDelta,
    AudioDelta,
    ImageDelta,
    ToolCallDelta,
    CitationDelta,
    ContinuationDelta,
    # auxiliary endpoints
    FileUploadRequest,
    FileInfo,
    FilePage,
    CacheInfo,
    CachePage,
    CachedPrefix,
    BatchRequest,
    BatchEntry,
    BatchJobInfo,
    VideoGenerationRequest,
    VideoJobInfo,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    AudioFormat,
    # live session types
    LiveConfig,
    # part factory for tool results (the one lowercase factory beginners
    # need at the top level; the rest stay in lm15.types)
    tool_result,
    # vocabulary aliases + constants
    Role,
    PartType,
    FinishReason,
    ReasoningEffort,
    ErrorCode,
    StreamEventType,
    ROLE_VALUES,
    FINISH_REASONS,
    ERROR_CODES,
)

# ── Access policies (auth by composition; spec/auth.md AUTH-10) ──────
from .features import AccessPolicy, EndpointSupport

# ── Errors ───────────────────────────────────────────────────────────
from .errors import (
    LM15Error,
    StreamAssemblyError,
    TransportError,
    ConfigurationError,
    CapabilityError,
    ProviderError,
    AuthError,
    BillingError,
    RateLimitError,
    InvalidRequestError,
    ContextLengthError,
    TimeoutError,
    ServerError,
    UnsupportedModelError,
    UnsupportedFeatureError,
    NotConfiguredError,
    RETRYABLE_ERRORS,
)

# ── Providers ────────────────────────────────────────────────────────
from .providers import (
    OpenAILM,
    OpenAIChatLM,
    AnthropicLM,
    GeminiLM,
    ClaudeCodeLM,
    OpenAICodexLM,
    XaiLM,
)
from .protocols import ProviderLM
from .providers.async_base import (
    AsyncOpenAILM,
    AsyncOpenAIChatLM,
    AsyncAnthropicLM,
    AsyncGeminiLM,
    AsyncClaudeCodeLM,
    AsyncOpenAICodexLM,
    AsyncXaiLM,
)

# ── Stream assembly ──────────────────────────────────────────────────
from .result import (
    ResponseStream,
    AsyncResponseStream,
    amaterialize_response,
    materialize_response,
    response_to_events,
)

# ── Model metadata (catalog hydration) ──────────────────────────────
from .models import ModelInfo, ModelRegistry

# ── Router (lm15.router) ─────────────────────────────────────────────
from .router import (
    AmbiguousModelError,
    AsyncLMRouter,
    LMRouter,
    MissingCredentialError,
    Resolution,
    RouterConfig,
    RouterError,
    UnknownModelError,
)

# ── Tool derivation (lm15.tools) ─────────────────────────────────────
from .tools import (
    DerivedParam,
    ToolConfig,
    ToolDerivation,
    ToolDerivationError,
    tool,
)
from .tools import derive as derive_tool

__all__ = [
    "__version__",
    # core
    "Request", "Response", "Message", "Usage", "Config", "CacheConfig",
    "Reasoning", "ToolChoice", "ErrorDetail", "ContinuationState",
    # parts
    "TextPart", "ThinkingPart", "RefusalPart", "CitationPart", "ImagePart",
    "AudioPart", "VideoPart", "DocumentPart", "BinaryPart", "ToolCallPart",
    "ToolResultPart",
    # tools
    "FunctionTool", "BuiltinTool", "ToolCallInfo", "tool_result",
    # logprobs
    "TokenLogprob", "TopLogprob",
    # streaming
    "StreamStartEvent", "StreamDeltaEvent", "StreamEndEvent",
    "StreamErrorEvent", "TextDelta", "ThinkingDelta", "AudioDelta",
    "ImageDelta", "ToolCallDelta", "CitationDelta", "ContinuationDelta",
    # auxiliary endpoints
    "FileUploadRequest", "FileInfo", "FilePage",
    "CacheInfo", "CachePage", "CachedPrefix",
    "BatchRequest", "BatchJobInfo", "BatchEntry",
    "VideoGenerationRequest", "VideoJobInfo", "VideoJob", "AsyncVideoJob",
    "BatchJob", "AsyncBatchJob",
    "ImageGenerationRequest", "ImageGenerationResponse",
    "SpeechGenerationRequest", "SpeechGenerationResponse", "AudioFormat",
    "LiveConfig", "Turn",
    # vocabularies
    "Role", "PartType", "FinishReason", "ReasoningEffort", "ErrorCode",
    "StreamEventType", "ROLE_VALUES", "FINISH_REASONS", "ERROR_CODES",
    # errors (the catchable taxonomy + the retry predicate)
    "LM15Error", "TransportError", "StreamAssemblyError", "ConfigurationError", "CapabilityError",
    "ProviderError", "AuthError", "BillingError", "RateLimitError",
    "InvalidRequestError", "ContextLengthError", "TimeoutError",
    "ServerError", "UnsupportedModelError", "UnsupportedFeatureError",
    "NotConfiguredError", "RETRYABLE_ERRORS",
    "AccessPolicy", "EndpointSupport",
    # providers
    "OpenAILM", "OpenAIChatLM", "AnthropicLM", "GeminiLM", "ClaudeCodeLM", "OpenAICodexLM", "XaiLM",
    "ProviderLM",
    # async mirror providers
    "AsyncOpenAILM", "AsyncOpenAIChatLM", "AsyncAnthropicLM", "AsyncGeminiLM",
    "AsyncClaudeCodeLM", "AsyncOpenAICodexLM", "AsyncXaiLM",
    # stream assembly
    "ResponseStream", "AsyncResponseStream",
    "materialize_response", "amaterialize_response", "response_to_events",
    # model metadata (catalog hydration)
    "ModelInfo", "ModelRegistry",
    # router (lm15.router; the rule/preset tables live there too)
    "LMRouter", "AsyncLMRouter", "RouterConfig", "Resolution",
    "RouterError", "UnknownModelError",
    "AmbiguousModelError", "MissingCredentialError",
    # tool derivation (lm15.tools)
    "tool", "derive_tool", "ToolConfig", "ToolDerivation", "DerivedParam",
    "ToolDerivationError",
]
