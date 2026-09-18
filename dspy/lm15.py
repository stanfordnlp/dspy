"""Use the lm15 types and providers bundled with DSPy.

Import public names here rather than from DSPy's private vendor directory.
These are the original lm15 objects, not copies or subclasses. `dspy.LM`
accepts explicit Request objects and uses them in its native engine path.

Examples:
    >>> from dspy.lm15 import Message, Request
    >>> request = Request(model="example", messages=(Message.user("Hello"),))
    >>> request.model
    'example'

This module also holds the providers declared to DSPy's native engine for
the process (`register_provider`). lm15 routes model strings through a
registry of receipted providers and lets a caller declare more with
`RouterConfig(providers=...)`. A registration is an environment fact, like
an API-key variable: a saved program records only the model string, and
loading it needs the same registration in place. Each `dspy.LM` binds the
registrations present when it is constructed and keeps them for its whole
life, so one LM never sees two definitions of a provider.
"""

import threading as _threading
from dataclasses import dataclass as _dataclass
from typing import Mapping as _Mapping

from dspy._vendor.lm15 import *
from dspy._vendor.lm15 import __all__ as _vendored_all
from dspy._vendor.lm15.compat import AnthropicCompat, OpenAIChatCompat, OpenAIResponsesCompat
from dspy._vendor.lm15.registry import ProviderDefinition
from dspy._vendor.lm15.router import RouterConfig

__all__ = [
    *_vendored_all,
    "AnthropicCompat",
    "ModelSupport",
    "OpenAIChatCompat",
    "OpenAIResponsesCompat",
    "ProviderDefinition",
    "RegisteredProvider",
    "register_provider",
    "registered_providers",
    "unregister_provider",
]


@_dataclass(frozen=True)
class ModelSupport:
    """What a model behind a declared provider supports, stated by the caller.

    A compat policy says what the wire accepts; whether a model honours it is
    model knowledge. None means "not stated": DSPy then reads the metadata
    snapshot through the provider's ``metadata_namespaces``, and treats what
    is still unknown as unsupported (the rule built-in providers live with).
    """

    function_calling: bool | None = None
    reasoning: bool | None = None
    response_schema: bool | None = None

    def __post_init__(self) -> None:
        for name in ("function_calling", "reasoning", "response_schema"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"ModelSupport.{name} is True, False or None, got {type(value).__name__}")

    def over(self, other: "ModelSupport | None") -> "ModelSupport":
        """This support, with unstated fields taken from ``other``."""
        if other is None:
            return self
        return ModelSupport(
            function_calling=self.function_calling if self.function_calling is not None else other.function_calling,
            reasoning=self.reasoning if self.reasoning is not None else other.reasoning,
            response_schema=self.response_schema if self.response_schema is not None else other.response_schema,
        )


@_dataclass(frozen=True)
class RegisteredProvider:
    """A declared provider as DSPy binds it: the lm15 definition (routing,
    credentials, wire) plus what DSPy alone needs — which metadata-snapshot
    namespaces may describe and price its models, and the caller's own
    statements of model support. Aliases on the definition are routing
    spellings only; they grant no metadata."""

    definition: ProviderDefinition
    metadata_namespaces: tuple[str, ...] = ()
    supports: ModelSupport | None = None
    models: tuple[tuple[str, ModelSupport], ...] = ()  # (model id, statement), immutable and picklable

    @property
    def id(self) -> str:
        return self.definition.id

    def support_for(self, model: str) -> ModelSupport:
        """The caller's statement for ``model``: its own entry over the
        provider-wide default; unstated fields stay None."""
        own = next((support for model_id, support in self.models if model_id == model), None)
        if own is None:
            return self.supports or ModelSupport()
        return own.over(self.supports)


_registry_lock = _threading.Lock()
_registered: dict[str, RegisteredProvider] = {}


def register_provider(
    definition: ProviderDefinition,
    *,
    supports: ModelSupport | None = None,
    models: _Mapping[str, ModelSupport] | None = None,
    metadata_namespaces: tuple[str, ...] = (),
    replace: bool = False,
) -> RegisteredProvider:
    """Declare a provider for the native ``dspy.LM``s constructed after this call.

    ``definition`` is an lm15 ``ProviderDefinition`` (``ProviderDefinition.chat(access,
    compat=...)`` for an OpenAI-compatible server). Afterwards
    ``dspy.LM("<id>/<model>")`` — and every alias — routes natively with the
    LM's ``api_key``, ``api_base`` and ``timeout`` honored, and the provider's
    own key variable read when none is given. A call the native route cannot
    carry falls back to LiteLLM at the declared address with the same
    credential, never at an address LiteLLM knows by a similar name.

    ``metadata_namespaces`` names the snapshot namespaces (LiteLLM's
    spellings, e.g. ``("fireworks_ai",)``) whose entries may describe and
    price this provider's models. Give it only when the endpoint really is
    that service; nothing is inherited from an alias. ``supports`` states
    model support for the whole provider and ``models`` per model id; a
    snapshot entry, when one is found, is the more specific fact and wins.

    Registering the same definition again with the same statements is a
    no-op. A different registration under the same id is refused unless
    ``replace=True``, and even then only LMs constructed afterwards bind it.
    A spelling another registered provider, lm15's registry or LiteLLM
    already uses is always refused.

    Returns:
        The registration as DSPy holds it.
    """
    if not isinstance(definition, ProviderDefinition):
        raise TypeError(
            "register_provider takes a dspy.lm15 ProviderDefinition; declare one with "
            f"ProviderDefinition.chat(access, compat=...), not {type(definition).__name__}"
        )
    if supports is not None and not isinstance(supports, ModelSupport):
        raise TypeError(f"supports= takes a dspy.lm15.ModelSupport, not {type(supports).__name__}")
    if not isinstance(metadata_namespaces, tuple) or not all(isinstance(n, str) and n for n in metadata_namespaces):
        raise TypeError("metadata_namespaces= is a tuple of non-empty snapshot namespace names")
    per_model: dict[str, ModelSupport] = {}
    for model_id, support in (models or {}).items():
        if not isinstance(model_id, str) or not model_id or not isinstance(support, ModelSupport):
            raise TypeError("models= maps model ids to dspy.lm15.ModelSupport")
        per_model[model_id] = support
    binding = RegisteredProvider(
        definition=definition,
        metadata_namespaces=metadata_namespaces,
        supports=supports,
        models=tuple(per_model.items()),
    )
    with _registry_lock:
        current = _registered.get(definition.id)
        if current is not None and current == binding:
            return current
        if current is not None and not replace:
            raise ValueError(
                f"provider {definition.id!r} is already registered with a different registration; "
                "pass replace=True to replace it (LMs constructed afterwards bind the replacement)"
            )
        for other in _registered.values():
            if other.id != definition.id and set(other.definition.spellings) & set(definition.spellings):
                clash = sorted(set(other.definition.spellings) & set(definition.spellings))
                raise ValueError(
                    f"provider {definition.id!r} spells {clash} like the registered provider "
                    f"{other.id!r}; unregister_provider({other.id!r}) first"
                )
        kept = {pid: b for pid, b in _registered.items() if pid != definition.id}
        # lm15 validates the set: type, hyphenation, and no spelling a
        # receipted entry or LiteLLM prefix already uses.
        RouterConfig(providers=(*(b.definition for b in kept.values()), definition))
        _registered.clear()
        _registered.update(kept)
        _registered[definition.id] = binding
        return binding


def unregister_provider(provider: str) -> None:
    """Forget a registered provider. Unknown ids are ignored. LMs that
    already bound it keep it."""
    with _registry_lock:
        _registered.pop(provider, None)


def registered_providers() -> tuple[RegisteredProvider, ...]:
    """The providers registered in this process, in registration order —
    the tuple a ``dspy.LM`` binds when constructed."""
    with _registry_lock:
        return tuple(_registered.values())


def _definitions(bindings) -> tuple[ProviderDefinition, ...]:
    """The lm15 definitions of an LM's bound providers, for RouterConfig."""
    return tuple(b.definition for b in bindings)


def _binding_for(bindings, provider: str) -> RegisteredProvider | None:
    """The LM's binding for a provider id, or None for a registry provider."""
    return next((b for b in bindings if b.id == provider), None)
