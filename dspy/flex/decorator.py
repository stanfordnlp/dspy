from __future__ import annotations

from typing import Any, Callable, overload

from dspy.flex.flex import Flex
from dspy.utils.annotation import experimental


@overload
def flex(signature_cls: type, /) -> Callable[..., Flex]: ...
@overload
def flex(
    *,
    persist_to: Any = None,
    context: Any = None,
    codegen_lm: Any = None,
    flex_id: str | None = None,
    auto_repair: bool = True,
) -> Callable[[type], Callable[..., Flex]]: ...


@experimental(version="3.3.0b2")
def flex(  # type: ignore[misc]
    signature_cls: type | None = None,
    *,
    persist_to: Any = None,
    context: Any = None,
    codegen_lm: Any = None,
    flex_id: str | None = None,
    auto_repair: bool = True,
):
    """Decorate a ``dspy.Signature`` subclass to produce a Flex module factory."""

    def _wrap(cls: type) -> Callable[..., Flex]:
        bound_persist_to = persist_to
        bound_context = context
        bound_codegen_lm = codegen_lm
        bound_flex_id = flex_id
        bound_auto_repair = auto_repair

        class FlexFactory:
            """Factory that instantiates a `dspy.Flex` bound to ``cls``."""

            signature = cls
            _signature_cls = cls
            _flex_persist_to = bound_persist_to
            _flex_context = bound_context
            _flex_codegen_lm = bound_codegen_lm
            _flex_id = bound_flex_id
            _flex_auto_repair = bound_auto_repair

            def __new__(
                cls,
                *,
                persist_to: Any = None,
                context: Any = None,
                codegen_lm: Any = None,
                flex_id: str | None = None,
                auto_repair: bool | None = None,
            ) -> Flex:
                return Flex(
                    signature=cls._signature_cls,
                    persist_to=persist_to if persist_to is not None else cls._flex_persist_to,
                    context=context if context is not None else cls._flex_context,
                    codegen_lm=codegen_lm if codegen_lm is not None else cls._flex_codegen_lm,
                    flex_id=flex_id if flex_id is not None else cls._flex_id,
                    auto_repair=auto_repair if auto_repair is not None else cls._flex_auto_repair,
                )

        FlexFactory.__name__ = getattr(cls, "__name__", "FlexFactory")
        FlexFactory.__qualname__ = FlexFactory.__name__
        FlexFactory.__doc__ = getattr(cls, "__doc__", None)
        return FlexFactory

    if signature_cls is not None:
        # Bare decorator form: @flex without parentheses.
        return _wrap(signature_cls)
    return _wrap
