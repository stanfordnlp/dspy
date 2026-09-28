"""
lm15.models — Optional model metadata and registry utilities.

The canonical inference request remains Request(model="..."). ModelInfo and
ModelRegistry are optional helpers for model discovery, validation, routing, and
cost estimation. Model capabilities are endpoint-specific (`inference` today)
so new endpoint families can be described later without breaking the
inference model. Fine-tune PROVENANCE stays describable through ModelOrigin
(`type`, `base_model`) — lm15 does inference with tuned models, not training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .compat import CompatProfile
from .types import JsonObject


# ─── Validation helpers ──────────────────────────────────────────────


def _check_nonempty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_positive_or_none(value: int | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{field_name} must be a positive integer or None")


def _check_non_negative_or_none(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number or None")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _coerce_float_field(obj: object, field_name: str) -> None:
    """Number rule (docs/serde-rules.md): float fields have ONE wire form.

    Same-valued int input is coerced (1 -> 1.0); bool never coerces.
    """
    value = getattr(obj, field_name)
    if type(value) is int:
        object.__setattr__(obj, field_name, float(value))


def _coerce_int_field(obj: object, field_name: str) -> None:
    """Number rule: int fields coerce same-valued floats, reject the rest."""
    value = getattr(obj, field_name)
    if type(value) is float:
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer")
        object.__setattr__(obj, field_name, int(value))


def _check_json_object_or_none(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object or None")


# ─── Pricing ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InferencePricing:
    input_per_million: float | None = None
    output_per_million: float | None = None
    cache_read_per_million: float | None = None
    cache_write_per_million: float | None = None
    currency: str = "USD"
    dimensions: JsonObject | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_per_million",
            "output_per_million",
            "cache_read_per_million",
            "cache_write_per_million",
        ):
            _check_non_negative_or_none(getattr(self, name), name)
            _coerce_float_field(self, name)
        _check_nonempty_text(self.currency, "currency")
        _check_json_object_or_none(self.dimensions, "dimensions")

    def estimate(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> float:
        """Estimate cost for the given token counts.

        ``None`` means "count unknown / not reported" (the ``Usage``
        zeros-vs-absent distinction) and is SKIPPED: an unknown dimension
        contributes nothing to the estimate, the same as a dimension whose
        rate is unset. It is NOT treated as zero tokens — the returned
        figure is a lower bound when any dimension is unknown. Pass an
        explicit ``0`` for a known-zero count.
        """
        total = 0.0
        if self.input_per_million is not None and input_tokens is not None:
            total += input_tokens * self.input_per_million / 1_000_000
        if self.output_per_million is not None and output_tokens is not None:
            total += output_tokens * self.output_per_million / 1_000_000
        if self.cache_read_per_million is not None and cache_read_tokens is not None:
            total += cache_read_tokens * self.cache_read_per_million / 1_000_000
        if self.cache_write_per_million is not None and cache_write_tokens is not None:
            total += cache_write_tokens * self.cache_write_per_million / 1_000_000
        return total


# ─── Endpoint-specific model capabilities ────────────────────────────


@dataclass(frozen=True, slots=True)
class InferenceModelInfo:
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_reasoning: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    pricing: InferencePricing | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_modalities", tuple(self.input_modalities))
        object.__setattr__(self, "output_modalities", tuple(self.output_modalities))
        object.__setattr__(self, "reasoning_efforts", tuple(self.reasoning_efforts))
        if any(not isinstance(v, str) or not v for v in self.input_modalities):
            raise ValueError("input_modalities must contain non-empty strings")
        if any(not isinstance(v, str) or not v for v in self.output_modalities):
            raise ValueError("output_modalities must contain non-empty strings")
        if any(not isinstance(v, str) or not v for v in self.reasoning_efforts):
            raise ValueError("reasoning_efforts must contain non-empty strings")
        _coerce_int_field(self, "context_window")
        _coerce_int_field(self, "max_output_tokens")
        _check_positive_or_none(self.context_window, "context_window")
        _check_positive_or_none(self.max_output_tokens, "max_output_tokens")
        _check_json_object_or_none(self.extensions, "extensions")


@dataclass(frozen=True, slots=True)
class ModelOrigin:
    type: str = "provider"
    id: str | None = None
    base_model: str | None = None
    provider_data: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_nonempty_text(self.type, "ModelOrigin.type")
        if self.id is not None:
            _check_nonempty_text(self.id, "ModelOrigin.id")
        if self.base_model is not None:
            _check_nonempty_text(self.base_model, "ModelOrigin.base_model")
        _check_json_object_or_none(self.provider_data, "provider_data")


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    provider: str
    api_family: str
    aliases: tuple[str, ...] = ()
    origin: ModelOrigin = field(default_factory=ModelOrigin)
    inference: InferenceModelInfo | None = None
    compat: CompatProfile | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_nonempty_text(self.id, "ModelInfo.id")
        _check_nonempty_text(self.provider, "ModelInfo.provider")
        _check_nonempty_text(self.api_family, "ModelInfo.api_family")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        if any(not isinstance(a, str) or not a for a in self.aliases):
            raise ValueError("ModelInfo.aliases must contain non-empty strings")
        if not isinstance(self.origin, ModelOrigin):
            raise TypeError("ModelInfo.origin must be a ModelOrigin")
        _check_json_object_or_none(self.extensions, "extensions")


# ─── Registry ────────────────────────────────────────────────────────


@dataclass(slots=True)
class ModelRegistry:
    _models: dict[tuple[str, str], ModelInfo] = field(default_factory=dict)
    _aliases: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)

    def add(self, model: ModelInfo, *, replace: bool = True) -> None:
        if not isinstance(model, ModelInfo):
            raise TypeError("model must be a ModelInfo")
        key = (model.provider, model.id)
        if not replace and key in self._models:
            raise ValueError(f"model already registered: {model.provider}/{model.id}")
        self._models[key] = model
        for alias in model.aliases:
            self._aliases[(model.provider, alias)] = key

    def add_profile(self, profile: object, *, replace: bool = True) -> None:
        models = getattr(profile, "models", None)
        if models is None:
            raise TypeError("profile must have a models attribute")
        for model in models:
            self.add(model, replace=replace)

    def get(self, provider: str, model: str) -> ModelInfo | None:
        key = (provider, model)
        if key in self._models:
            return self._models[key]
        target = self._aliases.get(key)
        if target is not None:
            return self._models.get(target)
        return None

    def resolve(self, model: str, provider: str | None = None) -> ModelInfo | None:
        if provider is not None:
            return self.get(provider, model)
        matches = [
            info
            for (_provider, model_id), info in self._models.items()
            if model_id == model or model in info.aliases
        ]
        return matches[0] if len(matches) == 1 else None

    def list(self, provider: str | None = None) -> tuple[ModelInfo, ...]:
        values = tuple(self._models.values())
        if provider is None:
            return values
        return tuple(m for m in values if m.provider == provider)

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted({provider for provider, _ in self._models}))

    @classmethod
    def from_profiles(cls, profiles: Iterable[object]) -> "ModelRegistry":
        registry = cls()
        for profile in profiles:
            registry.add_profile(profile)
        return registry

    @classmethod
    def from_dicts(cls, dicts: Iterable[dict]) -> "ModelRegistry":
        """Build a registry from canonical ModelInfo dicts.

        Each dict is validated through ``serde.model_info_from_dict``; the
        ModelInfo constructors reject junk (negative prices, empty modality
        names, missing id/provider) by raising ValueError/TypeError.
        """
        from .serde import model_info_from_dict

        registry = cls()
        for d in dicts:
            registry.add(model_info_from_dict(d))
        return registry

    @classmethod
    def discover(cls, *, group: str = "lm15.model_catalogs") -> "ModelRegistry":
        """Hydrate a registry from installed entry-point catalogs.

        Each entry point in ``group`` must load to a zero-argument callable
        returning an iterable of canonical ModelInfo dicts
        (docs/model-hydration.md). Catalogs are processed sorted by
        entry-point name; on duplicate (provider, id) keys the FIRST
        occurrence wins. A catalog that raises is skipped with a warning —
        discovery never crashes the host application.

        Hydrated data is ADVISORY metadata only: it must never change what
        build_request produces.
        """
        import warnings
        from importlib.metadata import entry_points

        from .serde import model_info_from_dict

        registry = cls()
        for ep in sorted(entry_points(group=group), key=lambda e: e.name):
            try:
                catalog = ep.load()
                models = [model_info_from_dict(d) for d in catalog()]
            except Exception as exc:
                warnings.warn(
                    f"lm15 model catalog {ep.name!r} failed and was skipped: {exc}",
                    stacklevel=2,
                )
                continue
            for model in models:
                if (model.provider, model.id) in registry._models:
                    continue  # first occurrence wins
                registry.add(model)
        return registry
