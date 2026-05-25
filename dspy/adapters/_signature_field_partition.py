"""Separate DSPy signature fields between adapter formatting and native LM channels."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, get_origin

from dspy.adapters.types import Type
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMToolSpec
from dspy.signatures.signature import Signature


@dataclass(frozen=True)
class _NativeResponseField:
    """A source signature output field filled from normalized LM response data."""

    name: str
    annotation: type[Type]


@dataclass
class _SignatureFieldPartition:
    """Result of separating one DSPy signature execution across LM channels.

    AI-engineering mental model:

    A DSPy signature is a source-level interface: it describes the semantic
    inputs and outputs of a module, not necessarily how every field should be
    transported to or from a model. Modern LMs expose more than a text prompt
    and a text completion: they have native tool specifications, tool-call
    outputs, reasoning content, citations, and eventually media/document input
    channels. Before rendering a request, DSPy must therefore separate the
    source signature fields into two paths:

    1. The remaining adapter format/parse path. These fields stay in
       `remaining_signature` / `remaining_inputs` and are passed to the
       adapter's ordinary `format()` method. Textual LM output for these fields
       is parsed with the same `remaining_signature` via `parse()`.

    2. Native LM request/response channels. These fields are removed from the
       remaining signature and represented explicitly as normalized request data
       (`tool_specs`, `request_kwargs`) or as source output fields that should be
       filled from `LMResponse` (`native_response_fields`).

    This object is not a planner or orchestrator. It is the value produced by a
    field-separation pass for one concrete signature execution, including the
    particular inputs and request kwargs for that execution. Keeping these
    values together makes the invariant visible: any field consumed by a native
    LM channel should no longer be rendered or parsed through the remaining
    adapter format.
    """

    source_signature: type[Signature]
    remaining_signature: type[Signature]
    remaining_inputs: dict[str, Any]
    request_kwargs: dict[str, Any]
    tool_specs: list[LMToolSpec] = dataclass_field(default_factory=list)
    native_response_fields: list[_NativeResponseField] = dataclass_field(default_factory=list)


def _partition_signature_fields(
    adapter: Any,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    inputs: dict[str, Any],
) -> _SignatureFieldPartition:
    """Separate signature fields between adapter format/parse and native LM channels."""
    partition = _SignatureFieldPartition(
        source_signature=signature,
        remaining_signature=signature,
        remaining_inputs=dict(inputs),
        request_kwargs=dict(lm_kwargs),
    )

    _partition_native_tool_calling(adapter, lm, partition)
    _partition_native_response_types(adapter, lm, partition)
    return partition


def _partition_native_tool_calling(adapter: Any, lm: BaseLM, partition: _SignatureFieldPartition) -> None:
    if not adapter.use_native_function_calling:
        return

    tool_call_input_field_name = _get_tool_call_input_field_name(partition.remaining_signature)
    tool_call_output_field_name = _get_tool_call_output_field_name(partition.remaining_signature)

    if tool_call_output_field_name and tool_call_input_field_name is None:
        raise ValueError(
            f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
            "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
            "input field with type `list[dspy.Tool]`."
        )

    if not (tool_call_output_field_name and lm.supports_function_calling):
        return

    tools = partition.remaining_inputs[tool_call_input_field_name]
    tools = tools if isinstance(tools, list) else [tools]
    partition.tool_specs.extend(_tool_to_lm_tool_spec(tool) for tool in tools)
    partition.native_response_fields.append(_NativeResponseField(tool_call_output_field_name, ToolCalls))
    partition.remaining_signature = partition.remaining_signature.delete(tool_call_output_field_name).delete(
        tool_call_input_field_name
    )
    partition.remaining_inputs.pop(tool_call_input_field_name, None)


def _partition_native_response_types(adapter: Any, lm: BaseLM, partition: _SignatureFieldPartition) -> None:
    for name, field in list(partition.remaining_signature.output_fields.items()):
        annotation = field.annotation
        if not (isinstance(annotation, type) and annotation in adapter.native_response_types and issubclass(annotation, Type)):
            continue
        if annotation is ToolCalls:
            continue

        before = partition.remaining_signature
        adapted = _adapt_native_response_type(annotation, partition.remaining_signature, name, lm, partition.request_kwargs)
        if adapted is not before:
            partition.native_response_fields.append(_NativeResponseField(name, annotation))
            partition.remaining_signature = adapted


def _adapt_native_response_type(
    annotation: type[Type],
    signature: type[Signature],
    field_name: str,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
) -> type[Signature]:
    # Keep existing Type-owned compatibility hooks for now. Later changes can
    # move built-in native behavior into explicit renderers/codecs while
    # retaining this as a fallback for user custom types.
    return annotation.adapt_to_native_lm_feature(signature, field_name, lm, lm_kwargs)


def _tool_to_lm_tool_spec(tool: Tool) -> LMToolSpec:
    args = tool.args or {}
    return LMToolSpec(
        name=tool.name or "",
        description=tool.desc,
        parameters={"type": "object", "properties": args, "required": list(args.keys())},
    )


def _get_tool_call_input_field_name(signature: type[Signature]) -> str | None:
    for name, field in signature.input_fields.items():
        origin = get_origin(field.annotation)
        if origin is list and field.annotation.__args__[0] == Tool:
            return name
        if field.annotation == Tool:
            return name
    return None


def _get_tool_call_output_field_name(signature: type[Signature]) -> str | None:
    for name, field in signature.output_fields.items():
        if field.annotation == ToolCalls:
            return name
    return None
