import re
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any

from litellm import ModelResponseStream

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
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

        self.adapter_identifiers = {
            "ChatAdapter": {
                "start_identifier": f"[[ ## {self.signature_field_name} ## ]]",
                "end_identifier": re.compile(r"\[\[ ## (\w+) ## \]\]"),
                "start_indicator": "[",
            },
            "JSONAdapter": {
                "start_identifier": f'"{self.signature_field_name}":',
                "end_identifier": re.compile(r"\w*\"(,|\s*})"),
                "start_indicator": '"',
            },
            "XMLAdapter": {
                "start_identifier": f"<{self.signature_field_name}>",
                "end_identifier": re.compile(rf"</{self.signature_field_name}>"),
                "start_indicator": "<",
            },
        }

    def _buffered_message_end_with_start_identifier(self, concat_message: str, start_identifier: str) -> str:
        for i in range(len(concat_message)):
            if start_identifier.startswith(concat_message[len(concat_message) - i - 1 :]):
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
                # Clear up the state for the next stream.
                self.stream_end = False
                self.cache_hit = False
                self.field_start_queue = []
                self.field_end_queue = Queue()
                self.stream_start = False
            else:
                return

        try:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is None:
                return
        except Exception:
            return

        if chunk_message and start_identifier in chunk_message:
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
            return

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
                chunk_message = concat_message[value_start_index:].lstrip()
                if isinstance(settings.adapter, JSONAdapter) and chunk_message.startswith('"'):
                    # For JSONAdapter, we need to remove the leading ". We cannot do this with the start_identifier
                    # because there could be a few splitters between ':' and '"', e.g., '"name": "value"'.
                    chunk_message = chunk_message[1:]

            elif self._buffered_message_end_with_start_identifier(concat_message.strip(), start_identifier):
                # If the buffered message ends with part of the start_identifier, we keep looking for the
                # start_identifier from the token stream.
                return
            else:
                # Doesn't match the expected identifier, reset the queue.
                self.field_start_queue = []
                return

        if self.stream_start:
            # The stream is started, we keep returning the token until we see the start of the next field.
            token = None
            self.field_end_queue.put(chunk_message)
            if self.field_end_queue.qsize() > 10:
                # We keep the last 10 tokens in the buffer to check if they form a valid identifier for end_identifier,
                # i.e., "[[ ## {next_field_name} ## ]]" for ChatAdapter to identify the end of the current field.
                # In most cases 10 tokens are enough to cover the end_identifier for all adapters.
                token = self.field_end_queue.get()
            concat_message = "".join(self.field_end_queue.queue).strip()
            if re.search(end_identifier, concat_message):
                # The next field is identified, we can end the stream and flush out all tokens in the buffer.
                self.stream_end = True
                last_token = self.flush()
                token = token + last_token if token else last_token
                token = token.rstrip()  # Remove the trailing \n\n

            if token:
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
            match = re.search(r'",|"\s*}', last_tokens)
            if match:
                boundary_index = match.start()
            else:
                boundary_index = len(last_tokens)
            return last_tokens[:boundary_index]
        elif isinstance(settings.adapter, XMLAdapter):
            boundary_index = last_tokens.find(f"</{self.signature_field_name}>")
            if boundary_index == -1:
                boundary_index = len(last_tokens)
            return last_tokens[:boundary_index]
        elif isinstance(settings.adapter, ChatAdapter) or settings.adapter is None:
            boundary_index = last_tokens.find("[[")
            return last_tokens[:boundary_index]
        else:
            raise ValueError(
                f"Unsupported adapter for streaming: {settings.adapter}, please use one of the following adapters: "
                f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
            )


def find_predictor_for_stream_listeners(program: "Module", stream_listeners: list[StreamListener]):
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
        for field_name, field_info in predictor.signature.output_fields.items():
            if field_name not in field_name_to_named_predictor:
                continue

            if field_name_to_named_predictor[field_name] is not None:
                raise ValueError(
                    f"Signature field {field_name} is not unique in the program, cannot automatically determine which "
                    "predictor to use for streaming. Please specify the predictor to listen to."
                )

            if field_info.annotation is not str:
                raise ValueError(
                    f"Stream listener can only be applied to string output field, but your field {field_name} is of "
                    f"type {field_info.annotation}."
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
