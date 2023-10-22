"""
Callback handler for emitting trace data in OpenInference tracing format.
OpenInference tracing is an open standard for capturing and storing
LLM Application execution logs.

It enables production LLMapp servers to seamlessly integrate with LLM
observability solutions such as Arize and Phoenix.

For more information on the specification, see
https://github.com/Arize-ai/open-inference-spec
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, TypedDict, cast)
from uuid import uuid4

from dspy.callbacks.base_handler import BaseCallbackHandler
from dspy.callbacks.schema import (TIMESTAMP_FORMAT, CBEvent, CBEventType,
                                   EventPayload)
from phoenix.trace.exporter import HttpExporter
from phoenix.trace.schemas import (Span, SpanEvent, SpanException, SpanID,
                                   SpanKind, SpanStatusCode)
from phoenix.trace.semantic_conventions import (
    DOCUMENT_CONTENT, DOCUMENT_ID, DOCUMENT_METADATA, DOCUMENT_SCORE,
    EMBEDDING_EMBEDDINGS, EMBEDDING_MODEL_NAME, EMBEDDING_TEXT,
    EMBEDDING_VECTOR, INPUT_MIME_TYPE, INPUT_VALUE, LLM_INPUT_MESSAGES,
    LLM_INVOCATION_PARAMETERS, LLM_MODEL_NAME, LLM_OUTPUT_MESSAGES,
    LLM_PROMPT_TEMPLATE, LLM_PROMPT_TEMPLATE_VARIABLES, LLM_PROMPTS,
    LLM_TOKEN_COUNT_COMPLETION, LLM_TOKEN_COUNT_PROMPT, LLM_TOKEN_COUNT_TOTAL,
    MESSAGE_CONTENT, MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
    MESSAGE_FUNCTION_CALL_NAME, MESSAGE_NAME, MESSAGE_ROLE, OUTPUT_MIME_TYPE,
    OUTPUT_VALUE, RERANKER_INPUT_DOCUMENTS, RERANKER_MODEL_NAME,
    RERANKER_OUTPUT_DOCUMENTS, RERANKER_QUERY, RERANKER_TOP_K,
    RETRIEVAL_DOCUMENTS, TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS,
    MimeType)
from phoenix.trace.tracer import SpanExporter, Tracer
from phoenix.trace.utils import get_stacktrace
from phoenix.utilities.error_handling import graceful_fallback

from dspy.internals.modules.lm import ChatMessage, ChatResponse

# from dspy.llms.base import ChatMessage, ChatResponse
# from llama_index.tools import ToolMetadata

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CBEventID = str
_LOCAL_TZINFO = datetime.now().astimezone().tzinfo


class CBEventData(TypedDict, total=False):
    name: str
    event_type: CBEventType
    start_event: CBEvent
    end_event: CBEvent
    attributes: Dict[str, Any]


ChildEventIds = Dict[CBEventID, List[CBEventID]]
EventData = Dict[CBEventID, CBEventData]


def payload_to_semantic_attributes(
    event_type: CBEventType,
    payload: Dict[str, Any],
    is_event_end: bool = False,
) -> Dict[str, Any]:
    """
    Converts a LLMapp payload to a dictionary of semantic conventions compliant attributes.
    """
    attributes: Dict[str, Any] = {}
    if event_type in (CBEventType.NODE_PARSING, CBEventType.CHUNKING):
        # TODO(maybe): handle these events
        return attributes
    if EventPayload.CHUNKS in payload and EventPayload.EMBEDDINGS in payload:
        attributes[EMBEDDING_EMBEDDINGS] = [
            {EMBEDDING_TEXT: text, EMBEDDING_VECTOR: vector}
            for text, vector in zip(payload[EventPayload.CHUNKS], payload[EventPayload.EMBEDDINGS])
        ]
    if event_type is not CBEventType.RERANKING and EventPayload.QUERY_STR in payload:
        attributes[INPUT_VALUE] = payload[EventPayload.QUERY_STR]
        attributes[INPUT_MIME_TYPE] = MimeType.TEXT
    if event_type is not CBEventType.RERANKING and EventPayload.NODES in payload:
        attributes[RETRIEVAL_DOCUMENTS] = [
            {
                DOCUMENT_ID: node_with_score.node.node_id,
                DOCUMENT_SCORE: node_with_score.score,
                DOCUMENT_CONTENT: node_with_score.node.text,
                DOCUMENT_METADATA: node_with_score.node.metadata,
            }
            for node_with_score in payload[EventPayload.NODES]
        ]
    if EventPayload.PROMPT in payload:
        attributes[LLM_PROMPTS] = [payload[EventPayload.PROMPT]]
    if EventPayload.MESSAGES in payload:
        messages = payload[EventPayload.MESSAGES]
        # Messages is only relevant to the LLM invocation
        if event_type is CBEventType.LLM:
            attributes[LLM_INPUT_MESSAGES] = [
                _message_payload_to_attributes(message_data) for message_data in messages
            ]
        elif event_type is CBEventType.AGENT_STEP and len(messages):
            # the agent step contains a message that is actually the input
            # akin to the query_str
            attributes[INPUT_VALUE] = _message_payload_to_str(messages[0])
    if response := (payload.get(EventPayload.RESPONSE) or payload.get(EventPayload.COMPLETION)):
        attributes.update(_get_response_output(response))
        if (raw := getattr(response, "raw", None)) is not None:
            attributes.update(_get_output_messages(raw))
            if (usage := getattr(raw, "usage", None)) is not None:
                attributes.update(_get_token_counts(usage))
    if event_type is CBEventType.RERANKING:
        if EventPayload.TOP_K in payload:
            attributes[RERANKER_TOP_K] = payload[EventPayload.TOP_K]
        if EventPayload.MODEL_NAME in payload:
            attributes[RERANKER_MODEL_NAME] = payload[EventPayload.MODEL_NAME]
        if EventPayload.QUERY_STR in payload:
            attributes[RERANKER_QUERY] = payload[EventPayload.QUERY_STR]
        if nodes := payload.get(EventPayload.NODES):
            attributes[RERANKER_OUTPUT_DOCUMENTS if is_event_end else RERANKER_INPUT_DOCUMENTS] = [
                {
                    DOCUMENT_ID: node_with_score.node.node_id,
                    DOCUMENT_SCORE: node_with_score.score,
                    DOCUMENT_CONTENT: node_with_score.node.text,
                    DOCUMENT_METADATA: node_with_score.node.metadata,
                }
                for node_with_score in nodes
            ]
    # if EventPayload.TOOL in payload:
    #     tool_metadata = cast(ToolMetadata, payload.get(EventPayload.TOOL))
    #     attributes[TOOL_NAME] = tool_metadata.name
    #     attributes[TOOL_DESCRIPTION] = tool_metadata.description
    #     attributes[TOOL_PARAMETERS] = tool_metadata.to_openai_function()["parameters"]
    if EventPayload.SERIALIZED in payload:
        serialized = payload[EventPayload.SERIALIZED]
        if event_type is CBEventType.EMBEDDING:
            if model_name := serialized.get("model_name"):
                attributes[EMBEDDING_MODEL_NAME] = model_name
        if event_type is CBEventType.LLM:
            if model_name := serialized.get("model"):
                attributes[LLM_MODEL_NAME] = model_name
                attributes[LLM_INVOCATION_PARAMETERS] = json.dumps(
                    {
                        "model": model_name,
                        "temperature": serialized["temperature"],
                        "max_tokens": serialized["max_tokens"],
                        **serialized["additional_kwargs"],
                    }
                )
    return attributes


class OpenInferenceTraceCallbackHandler(BaseCallbackHandler):
    """Callback handler for storing LLM application trace data in OpenInference format.
    OpenInference is an open standard for capturing and storing AI model
    inferences. It enables production LLMapp servers to seamlessly integrate
    with LLM observability solutions such as Arize and Phoenix.

    For more information on the specification, see
    https://github.com/Arize-ai/open-inference-spec
    """

    def __init__(
        self,
        callback: Optional[Callable[[List[Span]], None]] = None,
        exporter: Optional[SpanExporter] = None,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._tracer = Tracer(on_append=callback, exporter=exporter or HttpExporter())
        self._event_id_to_event_data: EventData = defaultdict(lambda: CBEventData())

    def _null_fallback(self, *args: Any, **kwargs: Any) -> None:
        return

    def _on_event_fallback(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: CBEventID = "",
        **kwargs: Any,
    ) -> CBEventID:
        return event_id or str(uuid4())

    @graceful_fallback(_on_event_fallback)
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: CBEventID = "",
        parent_id: CBEventID = "",
        **kwargs: Any,
    ) -> CBEventID:
        event_id = event_id or str(uuid4())
        event_data = self._event_id_to_event_data[event_id]
        event_data["name"] = event_type.value
        event_data["event_type"] = event_type
        event_data["start_event"] = CBEvent(
            event_type=event_type,
            payload=payload,
            id_=event_id,
        )
        event_data["attributes"] = {}
        # Parse the payload to extract the parameters
        if payload is not None:
            event_data["attributes"].update(
                payload_to_semantic_attributes(event_type, payload),
            )

        return event_id

    @graceful_fallback(_null_fallback)
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: CBEventID = "",
        **kwargs: Any,
    ) -> None:
        event_data = self._event_id_to_event_data[event_id]
        event_data.setdefault("name", event_type.value)
        event_data.setdefault("event_type", event_type)
        event_data["end_event"] = CBEvent(
            event_type=event_type,
            payload=payload,
            id_=event_id,
        )

        # Parse the payload to extract the parameters
        if payload is not None:
            event_data["attributes"].update(
                payload_to_semantic_attributes(event_type, payload, is_event_end=True),
            )

    @graceful_fallback(_null_fallback)
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._event_id_to_event_data = defaultdict(lambda: CBEventData())

    @graceful_fallback(_null_fallback)
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[ChildEventIds] = None,
    ) -> None:
        if not trace_map:
            return  # TODO: investigate when empty or None trace_map is passed
        _add_spans_to_tracer(
            event_id_to_event_data=self._event_id_to_event_data,
            trace_map=trace_map,
            tracer=self._tracer,
        )
        self._event_id_to_event_data = defaultdict(lambda: CBEventData())

    def get_spans(self) -> Iterator[Span]:
        """
        Returns the spans stored in the tracer. This is useful if you are running
        LlamaIndex in a notebook environment and you want to inspect the spans.
        """
        return self._tracer.get_spans()


def _add_spans_to_tracer(
    event_id_to_event_data: EventData,
    trace_map: ChildEventIds,
    tracer: Tracer,
) -> None:
    """
    Adds event data to the tracer, where it is converted to a span and stored in a buffer.

    Args:
        event_id_to_event_data (EventData): A map of event IDs to event data.

        trace_map (ChildEventIds): A map of parent event IDs to child event IDs. The root event IDs
        are stored under the key "root".

        tracer (Tracer): The tracer that stores spans.
    """

    trace_id = uuid4()
    parent_child_id_stack: List[Tuple[Optional[SpanID], CBEventID]] = [
        (None, root_event_id) for root_event_id in trace_map["root"]
    ]
    span_exceptions: List[SpanEvent] = []
    while parent_child_id_stack:
        parent_span_id, event_id = parent_child_id_stack.pop()
        event_data = event_id_to_event_data[event_id]
        event_type = event_data["event_type"]
        attributes = event_data["attributes"]
        if event_type is CBEventType.LLM:
            while parent_child_id_stack:
                preceding_event_parent_span_id, preceding_event_id = parent_child_id_stack[-1]
                if preceding_event_parent_span_id != parent_span_id:
                    break
                preceding_event_data = event_id_to_event_data[preceding_event_id]
                if preceding_event_data["event_type"] is not CBEventType.TEMPLATING:
                    break
                parent_child_id_stack.pop()
                if payload := preceding_event_data["start_event"].payload:
                    # Add template attributes to the LLM span to which they belong.
                    attributes.update(_template_attributes(payload))

        start_event = event_data["start_event"]
        start_time = _timestamp_to_tz_aware_datetime(start_event.time)
        if event_type is CBEventType.EXCEPTION:
            # LlamaIndex has exception callback events that are sibling events of the events in
            # which the exception occurred. We collect all the exception events and add them to the
            # relevant span.
            if (
                not start_event.payload
                or (error := start_event.payload.get(EventPayload.EXCEPTION)) is None
            ):
                continue
            span_exceptions.append(
                SpanException(
                    message=str(error),
                    timestamp=start_time,
                    exception_type=type(error).__name__,
                    exception_stacktrace=get_stacktrace(error),
                )
            )
            continue

        end_time = _get_end_time(event_data, span_exceptions)
        name = event_data["name"]
        span_kind = _get_span_kind(event_type)
        span = tracer.create_span(
            name=name,
            span_kind=span_kind,
            trace_id=trace_id,
            start_time=start_time,
            end_time=end_time,
            status_code=SpanStatusCode.ERROR if span_exceptions else SpanStatusCode.OK,
            status_message="",
            parent_id=parent_span_id,
            attributes=attributes,
            events=sorted(span_exceptions, key=lambda event: event.timestamp) or None,
            conversation=None,
        )
        span_exceptions = []
        new_parent_span_id = span.context.span_id
        for new_child_event_id in trace_map.get(event_id, []):
            parent_child_id_stack.append((new_parent_span_id, new_child_event_id))


def _get_span_kind(event_type: CBEventType) -> SpanKind:
    """Maps a CBEventType to a SpanKind.

    Args:
        event_type (CBEventType): LlamaIndex callback event type.

    Returns:
        SpanKind: The corresponding span kind.
    """
    return {
        CBEventType.EMBEDDING: SpanKind.EMBEDDING,
        CBEventType.LLM: SpanKind.LLM,
        CBEventType.RETRIEVE: SpanKind.RETRIEVER,
        CBEventType.FUNCTION_CALL: SpanKind.TOOL,
        CBEventType.AGENT_STEP: SpanKind.AGENT,
        CBEventType.RERANKING: SpanKind.RERANKER,
    }.get(event_type, SpanKind.CHAIN)


def _message_payload_to_attributes(message: Any) -> Dict[str, Optional[str]]:
    if isinstance(message, ChatMessage):
        message_attributes = {
            MESSAGE_ROLE: message.role.value,
            MESSAGE_CONTENT: message.content,
        }
        # Parse the kwargs to extract the function name and parameters for function calling
        # NB: these additional kwargs exist both for 'agent' and 'function' roles
        if "name" in message.additional_kwargs:
            message_attributes[MESSAGE_NAME] = message.additional_kwargs["name"]
        if "function_call" in message.additional_kwargs:
            function_call = message.additional_kwargs["function_call"]
            message_attributes[MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON] = function_call.arguments
            message_attributes[MESSAGE_FUNCTION_CALL_NAME] = function_call.name
        return message_attributes

    return {
        MESSAGE_ROLE: "user",  # assume user if not ChatMessage
        MESSAGE_CONTENT: str(message),
    }


def _message_payload_to_str(message: Any) -> Optional[str]:
    """Converts a message payload to a string, if possible"""
    if isinstance(message, ChatMessage):
        return message.content

    return str(message)


def _get_response_output(response: Any) -> Iterator[Tuple[str, Any]]:
    """
    Gets output from response objects. This is needed since the string representation of some
    response objects includes extra information in addition to the content itself. In the
    case of an agent's ChatResponse the output may be a `function_call` object specifying
    the name of the function to call and the arguments to call it with.
    """
    if isinstance(response, ChatResponse):
        message = response.message
        if content := message.content:
            yield OUTPUT_VALUE, content
            yield OUTPUT_MIME_TYPE, MimeType.TEXT
        else:
            yield OUTPUT_VALUE, json.dumps(message.additional_kwargs)
            yield OUTPUT_MIME_TYPE, MimeType.JSON
    else:
        yield OUTPUT_VALUE, str(response)
        yield OUTPUT_MIME_TYPE, MimeType.TEXT


def _get_end_time(event_data: CBEventData, span_events: List[SpanEvent]) -> Optional[datetime]:
    """
    A best-effort attempt to get the end time of an event.

    LlamaIndex's callback system does not guarantee that the on_event_end hook is always called, for
    example, when an error occurs mid-event.
    """
    if end_event := event_data.get("end_event"):
        tz_naive_end_time = _timestamp_to_tz_naive_datetime(end_event.time)
    elif span_events:
        last_span_event = sorted(span_events, key=lambda event: event.timestamp)[-1]
        tz_naive_end_time = last_span_event.timestamp
    else:
        return None
    return _tz_naive_to_tz_aware_datetime(tz_naive_end_time)


def _timestamp_to_tz_aware_datetime(timestamp: str) -> datetime:
    """Converts a timestamp string to a timezone-aware datetime."""
    return _tz_naive_to_tz_aware_datetime(_timestamp_to_tz_naive_datetime(timestamp))


def _timestamp_to_tz_naive_datetime(timestamp: str) -> datetime:
    """Converts a timestamp string to a timezone-naive datetime."""
    return datetime.strptime(timestamp, TIMESTAMP_FORMAT)


def _tz_naive_to_tz_aware_datetime(timestamp: datetime) -> datetime:
    """Converts a timezone-naive datetime to a timezone-aware datetime."""
    return timestamp.replace(tzinfo=_LOCAL_TZINFO)


def _get_message(message: object) -> Iterator[Tuple[str, Any]]:
    if role := getattr(message, "role", None):
        assert isinstance(role, str), f"content must be str, found {type(role)}"
        yield MESSAGE_ROLE, role
    if content := getattr(message, "content", None):
        assert isinstance(content, str), f"content must be str, found {type(content)}"
        yield MESSAGE_CONTENT, content
    if (function_call := getattr(message, "function_call", None)) is not None:
        if name := getattr(function_call, "name", None):
            assert isinstance(name, str), f"name must be str, found {type(name)}"
            yield MESSAGE_FUNCTION_CALL_NAME, name
        if arguments := getattr(function_call, "arguments", None):
            assert isinstance(arguments, str), f"arguments must be str, found {type(arguments)}"
            yield MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, arguments


def _get_output_messages(raw: object) -> Iterator[Tuple[str, Any]]:
    if not (choices := getattr(raw, "choices", None)):
        return
    assert isinstance(choices, Iterable), f"expected Iterable, found {type(choices)}"
    messages = [
        dict(_get_message(message))
        for choice in choices
        if (message := getattr(choice, "message", None)) is not None
    ]
    yield LLM_OUTPUT_MESSAGES, messages


def _get_token_counts(usage: object) -> Iterator[Tuple[str, Any]]:
    if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
    if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_TOTAL, total_tokens


def _template_attributes(payload: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Yields template attributes if present"""
    if template := payload.get(EventPayload.TEMPLATE):
        yield LLM_PROMPT_TEMPLATE, template
    if template_vars := payload.get(EventPayload.TEMPLATE_VARS):
        yield LLM_PROMPT_TEMPLATE_VARIABLES, template_vars
        # TODO(maybe): other keys in the same payload
        # EventPayload.SYSTEM_PROMPT
        # EventPayload.QUERY_WRAPPER_PROMPT
