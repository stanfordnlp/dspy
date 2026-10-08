from __future__ import annotations

import inspect
import json
import re
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Type
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import StreamResponse
from dspy.utils.lazy_import import require

jiter = require("jiter")
json_repair = require("json_repair")

if TYPE_CHECKING:
    from litellm import ModelResponseStream

    from dspy.primitives.module import Module

ADAPTER_SUPPORT_STREAMING = [ChatAdapter, XMLAdapter, JSONAdapter]


class StreamListener:
    """Class that listens to the stream to capture the streeaming of a specific output field of a predictor."""

    def __init__(
        self,
        signature_field_name: str,
        predict: Any = None,
        predict_name: str | None = None,
        allow_reuse: bool = False,
    ):
        """
        Args:
            signature_field_name: The name of the field to listen to.
            predict: The predictor to listen to. If None, when calling `streamify()` it will automatically look for
                the predictor that has the `signature_field_name` in its signature.
            predict_name: The name of the predictor to listen to. If None, when calling `streamify()` it will
                automatically look for the predictor that has the `signature_field_name` in its signature.
            allow_reuse: If True, the stream listener can be reused for multiple streams. Please note that this could
                hurt the performance because the same stream chunk is sent to multiple listeners.
        """
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name

        self.field_start_queue = []
        self.field_end_queue = Queue()
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False
        self.allow_reuse = allow_reuse

        self.json_adapter_state = {"field_accumulated_messages": "", "emitted_length": 0, "response_complete": False}
        self.value_started = False
        self.held_whitespace = ""
        self.xml_leading_whitespace = ""
        self.xml_has_child_markup = False

        self.adapter_identifiers = {
            "ChatAdapter": {
                "start_identifier": f"[[ ## {self.signature_field_name} ## ]]",
                "end_identifier": re.compile(r"\[\[ ## (\w+) ## \]\]"),
                "start_indicator": "[",
                "end_pattern_prefixes": ["[", "[[", "[[ ", "[[ #", "[[ ##"],
                "end_pattern_contains": "[[ ##",
            },
            "JSONAdapter": {
                "start_identifier": f'"{self.signature_field_name}":',
                "end_identifier": re.compile(r"\w*\"(,|\s*})"),
                "start_indicator": '"',
                "end_pattern_prefixes": ['"', '",', '" ', '"}'],
                "end_pattern_contains": "}",
            },
            "XMLAdapter": {
                "start_identifier": f"<{self.signature_field_name}>",
                "end_identifier": re.compile(rf"</{self.signature_field_name}>"),
                "start_indicator": "<",
                "end_pattern_prefixes": ["<", "</"],
                "end_pattern_contains": "</",  # Any closing tag start
            },
        }

    def _buffered_message_end_with_start_identifier(self, concat_message: str, start_identifier: str) -> str:
        for i in range(len(concat_message)):
            if start_identifier.startswith(concat_message[len(concat_message) - i - 1 :]):
                return True
        return False

    def _could_form_end_identifier(self, concat_message: str, adapter_name: str) -> bool:
        """Check if the buffered message could potentially form the end identifier.

        This prevents unnecessary buffering when the tokens clearly cannot form the end pattern.
        For example, if buffered message is "hello world" and end pattern is "[[ ## ... ## ]]",
        we know it cannot form the pattern, so we should yield immediately.

        Args:
            concat_message: The concatenated buffered message
            adapter_name: The name of the adapter being used

        Returns:
            True if the message could potentially form part of the end identifier
        """
        adapter_config = self.adapter_identifiers[adapter_name]
        end_pattern_prefixes = adapter_config.get("end_pattern_prefixes", [])
        end_pattern_contains = adapter_config.get("end_pattern_contains")

        # First check: does it end with a potential start of the pattern?
        if any(concat_message.endswith(prefix) for prefix in end_pattern_prefixes):
            return True

        # Second check: if there's a pattern marker, check if message contains it
        # This handles cases like "[[ ## com" where we have partial field name
        if end_pattern_contains and end_pattern_contains in concat_message:
            return True

        return False

    def receive(self, chunk: ModelResponseStream):
        adapter_name = settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter"
        if adapter_name not in self.adapter_identifiers:
            raise ValueError(
                f"Unsupported adapter for streaming: {adapter_name}, please use one of the following adapters: "
                f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
            )
        start_identifier = self.adapter_identifiers[adapter_name]["start_identifier"]
        end_identifier = self.adapter_identifiers[adapter_name]["end_identifier"]
        start_indicator = self.adapter_identifiers[adapter_name]["start_indicator"]

        if self.stream_end:
            if self.allow_reuse:
                if isinstance(settings.adapter, JSONAdapter) and self._output_type is str:
                    try:
                        message = chunk.choices[0].delta.content
                        finish_reason = chunk.choices[0].finish_reason
                    except Exception:
                        return
                    if not message:
                        if finish_reason is not None:
                            self.json_adapter_state["response_complete"] = True
                        return
                    if finish_reason is not None and not self.json_adapter_state["response_complete"]:
                        self.json_adapter_state["response_complete"] = True
                        return
                    if not self.json_adapter_state["response_complete"]:
                        try:
                            jiter.from_json(self.json_adapter_state["field_accumulated_messages"].encode("utf-8"))
                        except ValueError:
                            # This field ended, but the rest of its response is still arriving.
                            self.json_adapter_state["field_accumulated_messages"] += message
                            return
                        if not message.lstrip().startswith("{"):
                            return
                # Clear up the state for the next stream.
                self.stream_end = False
                self.cache_hit = False
                self.field_start_queue = []
                self.field_end_queue = Queue()
                self.json_adapter_state["field_accumulated_messages"] = ""
                self.json_adapter_state["emitted_length"] = 0
                self.json_adapter_state["response_complete"] = False
                self.stream_start = False
                self.value_started = False
                self.held_whitespace = ""
                self.xml_leading_whitespace = ""
                self.xml_has_child_markup = False
            else:
                return

        # Handle custom streamable types
        if (
            self._output_type
            and inspect.isclass(self._output_type)
            and issubclass(self._output_type, Type)
            and self._output_type.is_streamable()
        ):
            if parsed_chunk := self._output_type.parse_stream_chunk(chunk):
                return StreamResponse(
                    self.predict_name,
                    self.signature_field_name,
                    parsed_chunk,
                    is_last_chunk=self.stream_end,
                )

        # For non-custom streamable types, the streaming chunks come from the content field of the ModelResponseStream.
        try:
            chunk_message = chunk.choices[0].delta.content
            if not chunk_message:
                if (
                    isinstance(settings.adapter, JSONAdapter)
                    and self._output_type is str
                    and chunk.choices[0].finish_reason is not None
                ):
                    return self.finalize()
                return
            if (
                isinstance(settings.adapter, JSONAdapter)
                and self._output_type is str
                and chunk.choices[0].finish_reason is not None
            ):
                self.json_adapter_state["response_complete"] = True
        except Exception:
            return

        if chunk_message and start_identifier in chunk_message and not isinstance(settings.adapter, JSONAdapter):
            # If the cache is hit, the chunk_message could be the full response. When it happens we can
            # directly end the stream listening. In some models like gemini, each stream chunk can be multiple
            # tokens, so it's possible that response only has one chunk, we also fall back to this logic.
            message_after_start_identifier = chunk_message[
                chunk_message.find(start_identifier) + len(start_identifier) :
            ]
            if re.search(end_identifier, message_after_start_identifier):
                self.cache_hit = True
                self.stream_start = True
                self.stream_end = True
                return

        if len(self.field_start_queue) == 0 and not self.stream_start and start_indicator in chunk_message:
            # We look for the pattern of start_identifier, i.e., "[[ ## {self.signature_field_name} ## ]]" for
            # ChatAdapter to identify the start of the stream of our target field. Once the start_indicator, i.e., "[["
            # for ChatAdapter, is found, we start checking the next tokens
            self.field_start_queue.append(chunk_message)
            if (
                not isinstance(settings.adapter, JSONAdapter)
                or self._output_type is not str
                or start_identifier not in chunk_message
            ):
                return
            chunk_message = ""

        if len(self.field_start_queue) > 0 and not self.stream_start:
            # We keep appending the tokens to the queue until we have a full identifier or the concanated
            # tokens no longer match our expected identifier.
            self.field_start_queue.append(chunk_message)
            concat_message = "".join(self.field_start_queue)

            if start_identifier in concat_message:
                # We have a full identifier, we can start the stream.
                self.stream_start = True
                self.field_start_queue = []
                # Keep the part after the start_identifier from the concat_message, we need to write it to the buffer.
                value_start_index = concat_message.find(start_identifier) + len(start_identifier)
                chunk_message = concat_message[value_start_index:]
                if not isinstance(settings.adapter, XMLAdapter):
                    chunk_message = chunk_message.lstrip()

                if isinstance(settings.adapter, JSONAdapter):
                    # For JSONAdapter, we rely on partial json parsing to detect the end of the field we are listening
                    # to, so we need to maintain a few extra states to help us with that.
                    # We add an extra "{" to the beginning of the field_accumulated_messages, so we can detect the
                    # appearance of the next key.
                    self.json_adapter_state["field_accumulated_messages"] += "{" + start_identifier

            elif self._buffered_message_end_with_start_identifier(concat_message.strip(), start_identifier):
                # If the buffered message ends with part of the start_identifier, we keep looking for the
                # start_identifier from the token stream.
                return
            else:
                # Doesn't match the expected identifier, reset the queue.
                self.field_start_queue = []
                return

        if self.stream_start and chunk_message:
            if isinstance(settings.adapter, JSONAdapter) and self._output_type is str:
                if chunk.choices[0].finish_reason is not None:
                    self.json_adapter_state["field_accumulated_messages"] += chunk_message
                    return self.finalize()
                return self._json_adapter_handle_string_chunk(chunk_message)

            # The stream is started, we keep returning the token until we see the start of the next field.
            self.field_end_queue.put(chunk_message)

            token = None
            concat_message = "".join(self.field_end_queue.queue).strip()

            if not self._could_form_end_identifier(concat_message, adapter_name):
                # Buffer cannot form end identifier, safe to flush out the tokens in the buffer.
                token = self.flush()
            elif self.field_end_queue.qsize() > 10:
                # We keep the last 10 tokens in the buffer if they can potentially form the end_identifier to avoid
                # sending the DSPy boilerplate tokens to users. 10 is a heuristic number that is sufficient to capture
                # the end_identifier for all LMs.
                token = self.field_end_queue.get()

            # TODO: Put adapter streaming handling into individual classes, e.g., `JSONAdapterStreamListener`,
            # `ChatAdapterStreamListener`, `XMLAdapterStreamListener` instead of having many adhoc code in the
            # `StreamListener` class.
            if isinstance(settings.adapter, JSONAdapter):
                # JSONAdapter uses partial json parsing to detect the end of the field we are listening to, instead of
                # relying on the end_identifier.
                return self._json_adapter_handle_stream_chunk(token, chunk_message)
            else:
                # Other adapters rely on the end_identifier to detect the end of the field we are listening to.
                return self._default_handle_stream_chunk(token, end_identifier)

    def _json_adapter_handle_string_chunk(self, chunk_message: str) -> StreamResponse | None:
        self.json_adapter_state["field_accumulated_messages"] += chunk_message
        accumulated = self.json_adapter_state["field_accumulated_messages"].encode("utf-8")
        try:
            parsed = jiter.from_json(accumulated, partial_mode="trailing-strings")
            value = parsed.get(self.signature_field_name)
            if not isinstance(value, str):
                # JSONAdapter also accepts non-string JSON values in str fields.
                # Decode the complete value, then use the adapter's str coercion.
                start_identifier = self.adapter_identifiers["JSONAdapter"]["start_identifier"]
                value_source = self.json_adapter_state["field_accumulated_messages"][
                    len(start_identifier) + 1 :
                ].lstrip()
                value, end = json.JSONDecoder().raw_decode(value_source)
                if end == len(value_source) or value_source[end] not in " \t\r\n,}":
                    # A partial number like 4 may still become 42 or 4e2.
                    return None
                self.stream_end = True
                return StreamResponse(self.predict_name, self.signature_field_name, str(value), is_last_chunk=True)
            # Unlike trailing-strings mode, partial_mode=True omits unfinished strings.
            # A completed value ends this field even if the same chunk contains the next field.
            completed = jiter.from_json(accumulated, partial_mode=True)
        except ValueError:
            # An escape sequence may be split across provider chunks.
            return None

        self.stream_end = self.signature_field_name in completed
        token = value[self.json_adapter_state["emitted_length"] :]
        self.json_adapter_state["emitted_length"] = len(value)
        if token or self.stream_end:
            return StreamResponse(self.predict_name, self.signature_field_name, token, is_last_chunk=self.stream_end)

    def _json_adapter_handle_stream_chunk(self, token: str, chunk_message: str) -> StreamResponse | None:
        self.json_adapter_state["field_accumulated_messages"] += chunk_message
        if self.json_adapter_state["field_accumulated_messages"].rstrip().endswith("}"):
            # When the accumulated tokens end with a curly bracket, that means the streaming for the `dspy.Predict` we
            # are listening to is probably finished, we need to run a check and decide whether to end the stream.
            try:
                # If the parse doesn't raise an error, that means the accumulated tokens is a valid json object. Because
                # we add an extra "{" to the beginning of the field_accumulated_messages, so we know the streaming is
                # finished.
                jiter.from_json(self.json_adapter_state["field_accumulated_messages"].encode("utf-8"))
                self.stream_end = True
                last_token = self.flush()
                right_curly_bracket_index = last_token.rfind("}")
                token = (
                    token + last_token[:right_curly_bracket_index] if token else last_token[:right_curly_bracket_index]
                )
                return StreamResponse(
                    self.predict_name, self.signature_field_name, token, is_last_chunk=self.stream_end
                )
            except ValueError:
                pass

        try:
            parsed = jiter.from_json(
                self.json_adapter_state["field_accumulated_messages"].encode("utf-8"),
                partial_mode="trailing-strings",
            )
            if len(parsed) > 1:
                # If partial json parsing finds a second key, that means the streaming for the field we are listening to
                # is finished.
                self.stream_end = True
                last_token = self.flush()

                keys = list(parsed.keys())
                next_field_name = None
                for key in keys:
                    if key != self.signature_field_name:
                        next_field_name = key
                        break

                last_token_index = last_token.find(next_field_name)
                token = token + last_token[:last_token_index] if token else last_token[:last_token_index]
        except ValueError:
            pass

        if token or self.stream_end:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=self.stream_end,
            )

    def _default_handle_stream_chunk(self, token: str, end_identifier: str) -> StreamResponse | None:
        concat_message = "".join(self.field_end_queue.queue).strip()

        if re.search(end_identifier, concat_message):
            # The next field is identified, we can end the stream and flush out all tokens in the buffer.
            self.stream_end = True
            last_token = self.flush()
            token = token + last_token if token else last_token
            if not (isinstance(settings.adapter, XMLAdapter) and self._output_type is str):
                token = token.rstrip()  # Remove the trailing \n\n
        if isinstance(settings.adapter, XMLAdapter) and self._output_type is str:
            return self._xml_adapter_handle_string_chunk(token)

        # The parsed field value is stripped, so drop leading whitespace and hold back trailing whitespace until more
        # text arrives. Otherwise the newlines around the field headers leak into the chunks depending on how the
        # provider splits the stream.
        if token and not self.value_started:
            token = token.lstrip()
        if token:
            stripped = token.rstrip()
            if stripped:
                self.value_started = True
                token, self.held_whitespace = self.held_whitespace + stripped, token[len(stripped) :]
            else:
                self.held_whitespace += token
                token = ""

        if token or self.stream_end:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=self.stream_end,
            )

    def _xml_adapter_handle_string_chunk(self, token: str) -> StreamResponse | None:
        """Preserve raw XML string content when the value contains child markup."""
        if self.xml_leading_whitespace or (token and not self.value_started and token[0].isspace()):
            self.xml_leading_whitespace += token
            if re.search(r"<(?!/|\?|!)[A-Za-z_][^>]*>", self.xml_leading_whitespace):
                self.xml_has_child_markup = True
                token = self.xml_leading_whitespace
                self.xml_leading_whitespace = ""
            elif self.stream_end:
                token = self.xml_leading_whitespace.strip()
                self.xml_leading_whitespace = ""
            else:
                return None
        elif re.search(r"<(?!/|\?|!)[A-Za-z_][^>]*>", token):
            self.xml_has_child_markup = True
            token = self.held_whitespace + token
            self.held_whitespace = ""

        if self.xml_has_child_markup:
            if token or self.stream_end:
                return StreamResponse(
                    self.predict_name, self.signature_field_name, token, is_last_chunk=self.stream_end
                )
            return None

        # Childless XML strings are stripped by XMLAdapter. Match that behavior while
        # holding initial whitespace until we know whether it is meaningful XML text.
        if token and not self.value_started:
            token = token.lstrip()
        if token:
            stripped = token.rstrip()
            if stripped:
                self.value_started = True
                token, self.held_whitespace = self.held_whitespace + stripped, token[len(stripped) :]
            else:
                self.held_whitespace += token
                token = ""

        if token or self.stream_end:
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                token,
                is_last_chunk=self.stream_end,
            )

    def flush(self) -> str:
        """Flush all tokens in the field end queue.

        This method is called to flush out the last a few tokens when the stream is ended. These tokens
        are in the buffer because we don't directly yield the tokens received by the stream listener
        with the purpose to not yield the end_identifier tokens, e.g., "[[ ## ... ## ]]" for ChatAdapter.
        """
        last_tokens = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        if isinstance(settings.adapter, JSONAdapter):
            return last_tokens
        elif isinstance(settings.adapter, XMLAdapter):
            boundary_index = last_tokens.find(f"</{self.signature_field_name}>")
            if boundary_index == -1:
                boundary_index = len(last_tokens)
            return last_tokens[:boundary_index]
        elif isinstance(settings.adapter, ChatAdapter) or settings.adapter is None:
            boundary_index = last_tokens.find("[[")
            if boundary_index == -1:
                boundary_index = len(last_tokens)
            return last_tokens[:boundary_index]
        else:
            raise ValueError(
                f"Unsupported adapter for streaming: {settings.adapter}, please use one of the following adapters: "
                f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
            )

    def finalize(self) -> StreamResponse | None:
        """Finalize the stream and flush any remaining buffered tokens.

        This should be called when the stream ends.
        It ensures no tokens are lost from the buffer and marks the final chunk appropriately.

        Returns:
            A StreamResponse with the remaining buffered tokens and is_last_chunk=True,
            or None if there are no buffered tokens or the stream hasn't started.
        """
        if isinstance(settings.adapter, JSONAdapter) and self._output_type is str:
            # A top-level Prediction is also a response boundary for repaired JSON.
            self.json_adapter_state["response_complete"] = True
        if self.stream_end or not self.stream_start:
            # Stream already ended or never started, nothing to finalize
            return None

        self.stream_end = True
        if isinstance(settings.adapter, JSONAdapter) and self._output_type is str:
            # JSONAdapter accepts repaired/truncated JSON; drain the new string buffer
            # with the same repair and coercion rather than silently losing the field.
            parsed = json_repair.loads(self.json_adapter_state["field_accumulated_messages"])
            if isinstance(parsed, dict) and self.signature_field_name in parsed:
                value = str(parsed[self.signature_field_name])
                token = value[self.json_adapter_state["emitted_length"] :]
                return StreamResponse(self.predict_name, self.signature_field_name, token, is_last_chunk=True)
        if self.field_end_queue.qsize() > 0:
            token = self.flush()
            if token:
                token = self.held_whitespace + token
                return StreamResponse(
                    self.predict_name,
                    self.signature_field_name,
                    token,
                    is_last_chunk=True,
                )
        return None

    @property
    def _output_type(self) -> type | None:
        try:
            return self.predict.signature.output_fields[self.signature_field_name].annotation
        except Exception:
            return None


def find_predictor_for_stream_listeners(
    program: Module, stream_listeners: list[StreamListener]
) -> dict[int, list[StreamListener]]:
    """Find the predictor for each stream listener.

    This is a utility function to automatically find the predictor for each stream listener. It is used when some
    listeners don't specify the predictor they want to listen to. If a listener's `signature_field_name` is not
    unique in the program, this function will raise an error.
    """
    predictors = program.named_predictors()

    field_name_to_named_predictor = {}
    for listener in stream_listeners:
        if listener.predict:
            continue
        field_name_to_named_predictor[listener.signature_field_name] = None

    for name, predictor in predictors:
        for field_name in predictor.signature.output_fields:
            if field_name not in field_name_to_named_predictor:
                continue

            if field_name_to_named_predictor[field_name] is not None:
                raise ValueError(
                    f"Signature field {field_name} is not unique in the program, cannot automatically determine which "
                    "predictor to use for streaming. Please specify the predictor to listen to."
                )
            field_name_to_named_predictor[field_name] = (name, predictor)

    predict_id_to_listener = defaultdict(list)
    for listener in stream_listeners:
        if listener.predict:
            predict_id_to_listener[id(listener.predict)].append(listener)
            continue
        if listener.signature_field_name not in field_name_to_named_predictor:
            raise ValueError(
                f"Signature field {listener.signature_field_name} is not a field of any predictor in the program, "
                "cannot automatically determine which predictor to use for streaming. Please verify your field name or "
                "specify the predictor to listen to."
            )
        listener.predict_name, listener.predict = field_name_to_named_predictor[listener.signature_field_name]
        predict_id_to_listener[id(listener.predict)].append(listener)
    return predict_id_to_listener
