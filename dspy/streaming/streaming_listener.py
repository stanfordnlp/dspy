import re
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any, List

from litellm import ModelResponseStream

from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
    from dspy.primitives.program import Module


class StreamListener:
    """Class that listens to the stream to capture the streeaming of a specific output field of a predictor."""

    def __init__(self, signature_field_name: str, predict: Any = None, predict_name: str = None):
        """
        Args:
            signature_field_name: The name of the field to listen to.
            predict: The predictor to listen to. If None, when calling `streamify()` it will automatically look for
                the predictor that has the `signature_field_name` in its signature.
            predict_name: The name of the predictor to listen to. If None, when calling `streamify()` it will
                automatically look for the predictor that has the `signature_field_name` in its signature.
        """
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name

        self.field_start_queue = []
        self.field_end_queue = Queue()
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False

        self.start_identifier = f"[[ ## {self.signature_field_name} ## ]]"
        self.end_identifier = re.compile(r"\[\[ ## (\w+) ## \]\]")

    def receive(self, chunk: ModelResponseStream):
        if self.stream_end:
            return

        try:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is None:
                return
        except Exception:
            return

        if chunk_message and self.start_identifier in chunk_message:
            # If the cache is hit, the chunk_message could be the full response. When it happens we can
            # directly end the stream listening.
            self.cache_hit = True
            self.stream_start = True
            self.stream_end = True
            return

        if len(self.field_start_queue) == 0 and "[[" in chunk_message:
            # We look for the pattern "[[ ## {self.signature_field_name} ## ]]" to identify the start
            # of the stream of our target field. Once "[[" is found, we start checking the next tokens
            # to see if they match our expected identifier.
            self.field_start_queue.append(chunk_message)
            return

        if len(self.field_start_queue) > 0 and not self.stream_start:
            # We keep appending the tokens to the queue until we have a full identifier or the concanated
            # tokens no longer match our expected identifier.
            self.field_start_queue.append(chunk_message)
            concat_message = "".join(self.field_start_queue).strip()
            start_token_index = concat_message.find("[[")
            concat_message = concat_message[start_token_index:]
            if self.start_identifier == concat_message:
                # We have a full identifier, we can start the stream.
                self.stream_start = True
            elif self.start_identifier.startswith(concat_message):
                # The concanated tokens still match our expected identifier, we keep listening.
                return
            else:
                # Doesn't match the expected identifier, reset the queue.
                self.field_start_queue = []
        elif self.stream_start:
            # The stream is started, we keep returning the token until we see the start of the next field.
            token = None
            self.field_end_queue.put(chunk_message)
            if self.field_end_queue.qsize() > 10:
                # We keep the last 10 tokens in the buffer to check if they form a valid identifier for
                # "[[ ## {next_field_name} ## ]]" to identify the end of the current field. In most cases
                # 10 tokens are enough to cover "[[ ## {next_field_name} ## ]]".
                token = self.field_end_queue.get()
            concat_message = "".join(self.field_end_queue.queue).strip()
            if re.search(self.end_identifier, concat_message):
                # The next field is identified, we can end the stream and flush out all tokens in the buffer.
                self.stream_end = True
                last_token = self.flush()
                token = token + last_token if token else last_token
                token = token.rstrip()  # Remove the trailing \n\n

            if token:
                return StreamResponse(self.predict_name, self.signature_field_name, token)

    def flush(self) -> str:
        """Flush all tokens in the field end queue.

        This method is called to flush out the last a few tokens when the stream is ended. These tokens
        are in the buffer because we don't directly yield the tokens received by the stream listener
        with the purpose to not yield the tokens "[[ ## ... ## ]]".
        """
        last_tokens = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        start_token_index = last_tokens.find("[[")
        return last_tokens[:start_token_index]


def find_predictor_for_stream_listeners(program: "Module", stream_listeners: List[StreamListener]):
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

            if field_info.annotation != str:
                raise ValueError(
                    f"Stream listener can only be applied to string output field, but your field {field_name} is of "
                    f"type {field_info.annotation}."
                )

            field_name_to_named_predictor[field_name] = (name, predictor)

    predict_id_to_listener = defaultdict(list)
    for listener in stream_listeners:
        if listener.predict:
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
