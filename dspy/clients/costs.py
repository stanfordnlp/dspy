"""Token-cost estimation from provider-verbatim lm15 usage and upstream prices.

Unknown pricing is not zero. Hosted tools and unmodelled billing dimensions
return no estimate instead of claiming a complete bill from token prices alone.
"""

import re

from dspy.clients.model_metadata import model_info, rate, source_info


def estimate_cost(response, *, provider, requested_model, request=None):
    """Return (estimated USD cost, provenance), or (None, reason metadata).

    Advisory metadata must never turn a completed generation into a failure.
    """
    try:
        return _estimate_cost(response, provider=provider, requested_model=requested_model, request=request)
    except Warning:
        raise
    except Exception:
        return None, {"kind": "unknown", "reason": "pricing metadata could not be interpreted"}


def _estimate_cost(response, *, provider, requested_model, request=None):
    from dspy.lm15 import BuiltinTool

    def unknown(reason):
        return None, {"kind": "unknown", "reason": reason}

    if provider is None:
        return unknown("custom engine has no pricing provider")
    if any(part.type in {"image", "audio", "video", "document", "binary"} for part in response.message.parts):
        return unknown("generated-media billing is not covered by text token rates")
    if provider in {"claude-code", "openai-codex"}:
        return unknown("subscription usage is not per-token API billing")
    if request is not None and any(isinstance(tool, BuiltinTool) for tool in request.tools):
        return unknown("hosted-tool fees are not included in token rates")
    info = model_info(provider, response.model)
    priced_model = response.model
    if not info:
        info = model_info(provider, requested_model)
        priced_model = requested_model
    u = response.usage
    if u.input_tokens is None or u.output_tokens is None:
        return unknown("input or output usage was not reported")
    anthropic = provider in {"anthropic", "aws-anthropic", "bedrock-anthropic", "azure-anthropic",
                             "vertex-anthropic", "deepseek-anthropic", "meta-anthropic", "moonshotai-anthropic"}
    separate_reasoning = provider in {"gemini", "vertex", "vertex-express", "xai"}
    cached, written = u.cache_read_tokens or 0, u.cache_write_tokens or 0
    inputs = u.input_tokens if anthropic else u.input_tokens - cached - written
    reasoning = u.reasoning_tokens or 0
    outputs = u.output_tokens if separate_reasoning else u.output_tokens - reasoning
    input_audio, output_audio = u.input_audio_tokens or 0, u.output_audio_tokens or 0
    inputs -= input_audio
    outputs -= output_audio
    if min(inputs, outputs) < 0:
        return unknown("usage dimensions cannot be separated safely")
    total_input = u.input_tokens + cached + written if anthropic else u.input_tokens
    tier = request.config.service_tier if request is not None else None
    # Prefer the provider's actual served tier when it reports one.
    raw = response.provider_data or {}
    tier = raw.get("service_tier") or (raw.get("usageMetadata") or {}).get("serviceTier") or tier
    tier = {"default": None, "standard": None, "auto": None, "ON_DEMAND": None}.get(tier, tier)
    if tier not in (None, "priority", "flex"):
        return unknown("unrecognised service-tier pricing")
    suffix = f"_{tier}" if tier else ""
    thresholds = sorted({int(m.group(1)) * 1000 for key in info
                         if (m := re.search(r"_above_(\d+)k_tokens", key))})
    threshold = max((n for n in thresholds if total_input > n), default=None)
    long_context = f"_above_{threshold // 1000}k_tokens" if threshold else ""

    def price(key, *, fallback=None, long_write=False):
        duration = "_above_1hr" if long_write else ""
        candidate = key + duration + long_context + suffix
        found = rate(info, candidate)
        if found is not None:
            return found
        # Reasoning/audio/cache use their own rate where supplied; otherwise
        # use the matching tier's ordinary token rate, never a cheaper tier.
        if fallback and not duration:
            found = rate(info, fallback + long_context + suffix)
            if found is not None:
                return found
        return None

    long_write = bool(request is not None and request.config.cache is not None
                      and request.config.cache.retention == "long" and anthropic)
    dimensions = [
        (inputs, price("input_cost_per_token")),
        (outputs, price("output_cost_per_token")),
        (reasoning, price("output_cost_per_reasoning_token", fallback="output_cost_per_token")),
        (cached, price("cache_read_input_token_cost", fallback="input_cost_per_token")),
        (written, price("cache_creation_input_token_cost", fallback="input_cost_per_token", long_write=long_write)),
        (input_audio, price("input_cost_per_audio_token", fallback="input_cost_per_token")),
        (output_audio, price("output_cost_per_audio_token", fallback="output_cost_per_token")),
    ]
    if any(count and amount is None for count, amount in dimensions):
        return unknown("a billed token dimension has no known rate")
    if not info:
        return unknown("no matching pricing entry")
    return sum(count * amount for count, amount in dimensions if count and amount is not None), {
        "kind": "estimate", "currency": "USD", "provider": provider, "model": priced_model,
        "metadata": source_info(),
    }
