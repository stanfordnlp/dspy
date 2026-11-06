"""The provider tables, as data, for the vet protocol's ``provider_tables`` op.

``lm15-contract/tables/providers.json`` is this function's output, and
``tools/audit.py`` there fails when the two differ. Ports generate their
copies from that file (``playbooks/port.md`` rule 2: tables are data, copied,
never re-derived). This module reads the tables the reference uses at import
time and serializes them; it adds no fact of its own.

Shape rules, so a generator never guesses:

- Compat policies list only the knobs a preset sets (``None`` means
  "inherit", so an absent key is exactly the value), with
  ``model_overrides`` as ``[prefix, {knob: value}]`` pairs in match order.
- Access policies, host specs and endpoint support are written whole, every
  field, so a port never depends on its own defaults agreeing with these.
- Tuples become arrays and keep their order; mappings keep insertion order;
  a frozenset (``EndpointSupport.extra``) becomes a sorted array.
- Field names are the reference's (snake_case); a port renames them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

SCHEMA = 1


_NO_DEFAULT = object()


def _default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return _NO_DEFAULT


def _plain(value: Any, *, elide: tuple[type, ...]) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            if isinstance(value, elide) and v == _default(f):
                continue  # a compat knob at its default inherits
            out[f.name] = _plain(v, elide=elide)
        return out
    if isinstance(value, Mapping):
        return {str(k): _plain(v, elide=elide) for k, v in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(_plain(v, elide=elide) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v, elide=elide) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"provider_tables: no JSON form for {type(value).__name__}")


def provider_tables() -> dict[str, Any]:
    from . import compat as _compat
    from .login import declared as _declared
    from .login import flows as _flows
    from . import registry as _registry
    from . import router as _router

    elide = (_compat.OpenAIChatCompat, _compat.OpenAIResponsesCompat, _compat.AnthropicCompat)

    def plain(value: Any) -> Any:
        return _plain(value, elide=elide)

    def row(definition: Any) -> dict[str, Any]:
        compat = definition.compat
        return {
            "id": definition.id,
            "dialect": definition.dialect,
            "kind": "hosted" if definition.hosted else ("bound" if definition.bound else "adapter-owned"),
            "compat": compat if compat is None or isinstance(compat, str) else plain(compat),
            "access": plain(definition.access),
            "aliases": list(definition.aliases),
            "placeholder_key": definition.placeholder_key,
            "console_url": definition.console_url,
            "note": definition.note,
        }

    return {
        "schema": SCHEMA,
        "providers": [row(d) for d in _registry.PROVIDERS.values()],
        "declared_login": [row(d) for d in _declared.DECLARED_PROVIDERS],
        "compat": {
            "chat": plain(_compat.OPENAI_CHAT_PRESETS),
            "chat_base_urls": plain(_compat.OPENAI_CHAT_PRESET_BASE_URLS),
            "responses": plain(_compat.OPENAI_RESPONSES_PRESETS),
            "responses_base_urls": plain(_compat.OPENAI_RESPONSES_PRESET_BASE_URLS),
            "anthropic": plain(_compat.ANTHROPIC_PRESETS),
            "anthropic_base_urls": plain(_compat.ANTHROPIC_PRESET_BASE_URLS),
            "preset_aliases": plain(_compat._OPENAI_CHAT_PRESET_ALIASES),
        },
        "routing": {
            "default_rules": [{"prefix": r.prefix, "provider": r.provider, "note": r.note} for r in _router.DEFAULT_RULES],
            "litellm_prefixes": plain(_router.LITELLM_PROVIDER_PREFIXES),
        },
        "login": {"service_labels": plain(_flows._SERVICE_LABELS)},
    }
