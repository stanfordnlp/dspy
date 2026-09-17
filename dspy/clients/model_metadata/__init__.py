"""Model metadata without an inference SDK dependency.

Match the existing remote-first loading policy: once on first use, fetch the
upstream map with a five-second timeout, otherwise use the packaged backup.
LITELLM_LOCAL_MODEL_COST_MAP and LITELLM_MODEL_COST_MAP_URL retain their meaning.
No persistent remote cache is introduced. Metadata is advisory, not routing.
"""

import copy
import gzip
import json
import logging
import math
import os
import re
import threading
import urllib.request
from importlib.resources import files

logger = logging.getLogger(__name__)
_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_RESERVED = {"sample_spec", "fallback_generalizations"}
_lock = threading.RLock()
_data = None
_source = {}


def _snapshot():
    return json.loads(gzip.decompress(files(__package__).joinpath("snapshot.json.gz").read_bytes()))


def _count(data):
    return sum(key not in _RESERVED and isinstance(value, dict) for key, value in data.items())


async def apreload():
    """Load once without blocking an event loop (including lock contention)."""
    if _data is None:
        import asyncio

        await asyncio.to_thread(_load)


def _load():
    global _data, _source
    # The map is published only after it is fully built. Normal reads need
    # neither a network call nor a blocking lock after initialization.
    if _data is not None:
        return _data
    with _lock:
        if _data is not None:
            return _data
        local = _snapshot()
        forced = os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
        url = os.getenv("LITELLM_MODEL_COST_MAP_URL", _URL)
        data = local
        reason = None
        remote = False
        if not forced:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "DSPy/model-metadata"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = response.read(32 * 1024 * 1024 + 1)
                if len(payload) > 32 * 1024 * 1024:
                    raise ValueError("metadata exceeds 32 MB")
                candidate = json.loads(payload)
                if not isinstance(candidate, dict) or _count(candidate) < max(100, _count(local) * 0.5):
                    raise ValueError("metadata is empty or has shrunk unexpectedly")
                if any(not isinstance(v, dict) for k, v in candidate.items() if k not in _RESERVED):
                    raise ValueError("model entries must be objects")
                data, remote = candidate, True
            except Exception as exc:
                # The configured URL may contain credentials; do not log it or
                # exceptions from the HTTP client verbatim.
                reason = type(exc).__name__
                logger.warning("Model metadata refresh failed (%s); using bundled snapshot.", reason)
        _source = {"source": "remote" if remote else "local", "is_env_forced": forced,
                   "fallback_reason": reason,
                   "snapshot_version": json.loads(files(__package__).joinpath("provenance.json").read_text())["version"]}
        _data = data
        return data


def source_info():
    """Describe the loaded map without exposing configured URL credentials."""
    _load()
    return dict(_source)


def _provider_names(provider):
    # Metadata namespaces, NOT inference routes. Cloud deployment prices must
    # not fall back to a public provider's prices.
    return {
        "openai-chat": ("openai",), "claude-code": ("anthropic",),
        "openai-codex": ("openai",), "vertex": ("vertex_ai", "gemini"),
        "vertex-express": ("vertex_ai", "gemini"), "vertex-anthropic": ("vertex_ai",),
        "azure-chat": ("azure",), "azure-anthropic": ("azure_ai",),
        "bedrock-chat": ("bedrock",), "bedrock-mantle-chat": ("bedrock_mantle",),
        "bedrock-anthropic": ("bedrock",), "aws-anthropic": ("aws-anthropic",),
        "deepseek-anthropic": ("deepseek",), "meta-chat": ("meta",),
        "meta-anthropic": ("meta",), "moonshotai": ("moonshot", "moonshotai"),
        "moonshotai-responses": ("moonshot", "moonshotai"),
        "moonshotai-anthropic": ("moonshot", "moonshotai"),
        "vllm": ("hosted_vllm", "vllm"),
    }.get(provider, (provider,))


def model_info(provider, model):
    """Return a copy of the most specific matching entry, or an empty dict.

    Missing capability fields may inherit from a provider-matched bare entry;
    explicit false values do not. Prices never inherit from another entry.
    """
    data = _load()
    names = _provider_names(provider)
    keys = [f"{name}/{model}" for name in names]
    bare = data.get(model)
    if isinstance(bare, dict) and bare.get("litellm_provider") in names:
        keys.append(model)
    found = next((data[key] for key in keys if key in data and key not in _RESERVED), None)
    if found is None:
        for key, entry in data.items():
            if key in _RESERVED or entry.get("litellm_provider") not in names:
                continue
            aliases = entry.get("aliases")
            if isinstance(aliases, list) and any(alias in keys or alias == model for alias in aliases):
                found = entry
                break
    result = copy.deepcopy(found or {})
    if isinstance(bare, dict) and bare.get("litellm_provider") in names:
        for key, value in bare.items():
            if key.startswith("supports_") and result.get(key) is None:
                result[key] = value
    # Upstream family declarations supply capabilities, never guessed pricing.
    rules = data.get("fallback_generalizations", {}).get("rules", [])
    for rule in rules if isinstance(rules, list) else []:
        if not isinstance(rule, dict) or not isinstance(rule.get("pattern"), str):
            continue
        try:
            matches = re.search(rule["pattern"], model) is not None
        except re.error:
            continue
        if matches:
            for key, value in (rule.get("model_info") or {}).items():
                if key.startswith("supports_") and result.get(key) is None:
                    result[key] = value
    return result


def rate(info, key):
    value = info.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
