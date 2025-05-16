from typing import TYPE_CHECKING, Any, Optional, Type, get_args

from dspy.adapters.types import History
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class Adapter:
    def __init__(self, callbacks: Optional[list[BaseCallback]] = None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def _call_post_process(self, outputs: list[dict[str, Any]], signature: Type[Signature]) -> list[dict[str, Any]]:
        values = []

        for output in outputs:
            output_logprobs = None

            if isinstance(output, dict):
                output, output_logprobs = output["text"], output["logprobs"]

            value = self.parse(signature, output)

            if output_logprobs is not None:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values

    @staticmethod
    def _has_custom_format(val: Any, annotation: Any) -> bool:
        args = get_args(annotation)
        if hasattr(val, '_format') or hasattr(annotation, '_format'):
            return True
        if args and any(hasattr(arg, '_format') for arg in args):
            return True
        if hasattr(annotation, '__annotations__'):
            for field_name in annotation.__annotations__:
                field_val = val.get(field_name) if isinstance(val, dict) else getattr(val, field_name, None)
                field_ann = annotation.__annotations__.get(field_name)
                if Adapter._has_custom_format(field_val, field_ann):
                    return True
        return False
    
    @staticmethod
    def _preprocess_inputs_for_custom_format(signature, inputs: dict[str, Any]) -> dict[str, Any]:
        processed = {}
        for k, _ in signature.input_fields.items():
            v = inputs.get(k)
            annotation = signature.input_fields.get(k).annotation
            if Adapter._has_custom_format(v, annotation):
                processed[k] = f"<<CUSTOM_TYPE_TAG::{k}>>"
            else:
                processed[k] = v
        return processed
    
    @staticmethod
    def _custom_format_messages(messages: list[dict], signature, original_inputs: dict[str, Any]) -> list[dict]:
        new_messages = []
        for msg in messages:
            if msg.get("role") != "user":
                new_messages.append(msg)
                continue
            has_custom_type_tag = any(
                Adapter._has_custom_format(original_inputs.get(k), field.annotation)
                for k, field in signature.input_fields.items()
            )
            if not has_custom_type_tag:
                new_messages.append(msg)
                continue
            content_blocks = []
            current_content = msg.get("content")
            for k, _ in signature.input_fields.items():
                val = original_inputs.get(k)
                annotation = signature.input_fields[k].annotation
                args = get_args(annotation)
                custom_format_output = None
                if hasattr(val, '_format'):
                    custom_format_output = val._format()
                elif hasattr(annotation, '_format'):
                    instance = annotation.model_validate(val)
                    custom_format_output = instance._format()
                elif args and all(hasattr(arg, '_format') for arg in args):
                    custom_format_output = []
                    inner_cls = args[0]
                    for item in val:
                        instance = item if isinstance(item, inner_cls) else inner_cls.model_validate(item)
                        custom_format_output.extend(instance._format())
                elif hasattr(annotation, '__annotations__'):
                    cls = annotation
                    instance = cls.model_validate(val) if isinstance(val, dict) else val
                    custom_format_output = []
                    for field_name, field_ann in cls.__annotations__.items():
                        field_val = getattr(instance, field_name, None)
                        field_args = get_args(field_ann)
                        if hasattr(field_val, '_format'):
                            custom_format_output.extend(field_val._format())
                        elif hasattr(field_ann, '_format'):
                            inner_instance = field_ann.model_validate(field_val)
                            custom_format_output.extend(inner_instance._format())
                        elif any(
                            hasattr(nested_arg, '_format')
                            for arg in field_args if arg is not type(None)
                            for nested_arg in get_args(arg) or [arg]
                        ):
                            inner_cls = field_args[0]
                            inner_args = get_args(inner_cls) or [inner_cls]
                            actual_cls = inner_args[0]
                            for item in field_val:
                                inner_instance = item if isinstance(item, actual_cls) else actual_cls.model_validate(item)
                                custom_format_output.extend(inner_instance._format())
                if custom_format_output:
                    custom_type_tag = f"<<CUSTOM_TYPE_TAG::{k}>>"
                    if custom_type_tag in current_content:
                        before_tag, after_tag = current_content.split(custom_type_tag, 1)
                        if before_tag.strip():
                            content_blocks.append({"type": "text", "text": before_tag.strip()})
                        content_blocks.extend(custom_format_output)
                        current_content = after_tag
            if current_content.strip():
                content_blocks.append({"type": "text", "text": current_content.strip()})
            new_messages.append({"role": msg["role"], "content": content_blocks})
        return new_messages

    def __call__(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)

        outputs = lm(messages=inputs, **lm_kwargs)
        return self._call_post_process(outputs, signature)

    async def acall(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)

        outputs = await lm.acall(messages=inputs, **lm_kwargs)
        return self._call_post_process(outputs, signature)

    def format(
        self,
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the input messages for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
        the formatting of the input messages.

        In general we recommend the messages to have the following structure:
        ```
        [
            {"role": "system", "content": system_message},
            # Begin few-shot examples
            {"role": "user", "content": few_shot_example_1_input},
            {"role": "assistant", "content": few_shot_example_1_output},
            {"role": "user", "content": few_shot_example_2_input},
            {"role": "assistant", "content": few_shot_example_2_output},
            ...
            # End few-shot examples
            # Begin conversation history
            {"role": "user", "content": conversation_history_1_input},
            {"role": "assistant", "content": conversation_history_1_output},
            {"role": "user", "content": conversation_history_2_input},
            {"role": "assistant", "content": conversation_history_2_output},
            ...
            # End conversation history
            {"role": "user", "content": current_input},
        ]

        And system message should contain the field description, field structure, and task description.
        ```


        Args:
            signature: The DSPy signature for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        original_inputs = dict(inputs)

        # If the signature and inputs have conversation history, we need to format the conversation history and
        # remove the history field from the signature.
        history_field_name = self._get_history_field_name(signature, original_inputs)
        if history_field_name:
            # In order to format the conversation history, we need to remove the history field from the signature.
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                original_inputs,
            )

        messages = []
        system_message = (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        messages_after_demos = []
        inputs_copy = self._preprocess_inputs_for_custom_format(signature, original_inputs)
        if history_field_name:
            # Conversation history and current input
            content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
            messages_after_demos.extend(conversation_history)
            messages_after_demos.append({"role": "user", "content": content})
        else:
            # Only current input
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
            messages_after_demos.append({"role": "user", "content": content})
        messages = messages + self._custom_format_messages(messages_after_demos, signature, original_inputs)
        return messages

    def format_field_description(self, signature: Type[Signature]) -> str:
        """Format the field description for the system message.

        This method formats the field description for the system message. It should return a string that contains
        the field description for the input fields and the output fields.

        Args:
            signature: The DSPy signature for which to format the field description.

        Returns:
            A string that contains the field description for the input fields and the output fields.
        """
        raise NotImplementedError

    def format_field_structure(self, signature: Type[Signature]) -> str:
        """Format the field structure for the system message.

        This method formats the field structure for the system message. It should return a string that dictates the
        format the input fields should be provided to the LM, and the format the output fields will be in the response.
        Refer to the ChatAdapter and JsonAdapter for an example.

        Args:
            signature: The DSPy signature for which to format the field structure.
        """
        raise NotImplementedError

    def format_task_description(self, signature: Type[Signature]) -> str:
        """Format the task description for the system message.

        This method formats the task description for the system message. In most cases this is just a thin wrapper
        over `signature.instructions`.

        Args:
            signature: The DSPy signature of the DSpy module.

        Returns:
            A string that describes the task.
        """
        raise NotImplementedError

    def format_user_message_content(
        self,
        signature: Type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            signature: The DSPy signature for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            A string that contains the user message content.
        """
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        signature: Type[Signature],
        outputs: dict[str, Any],
        missing_field_message: Optional[str] = None,
    ) -> str:
        """Format the assistant message content.

        This method formats the assistant message content, which can be used in formatting few-shot examples,
        conversation history.

        Args:
            signature: The DSPy signature for which to format the assistant message content.
            outputs: The output fields to be formatted.
            missing_field_message: A message to be used when a field is missing.

        Returns:
            A string that contains the assistant message content.
        """
        raise NotImplementedError

    def format_demos(self, signature: Type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the signature.

        Returns:
            A list of multiturn messages.
        """
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            # Check if all fields are present and not None
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                # We only keep incomplete demos that have at least one input and one output field
                incomplete_demos.append(demo)

        messages = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            demo_copy = self._preprocess_inputs_for_custom_format(signature, demo)
            user_msg = {
                "role": "user",
                "content": self.format_user_message_content(signature, demo_copy, prefix=incomplete_demo_prefix),
            }
            messages.append(self._custom_format_messages([user_msg], signature, demo))
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this particular example. "
                    ),
                }
            )

        for demo in complete_demos:
            demo_copy = self._preprocess_inputs_for_custom_format(signature, demo)
            user_msg = {
                "role": "user",
                "content": self.format_user_message_content(signature, demo_copy),
            }
            messages= self._custom_format_messages([user_msg], signature, demo)
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this conversation history message. "
                    ),
                }
            )

        return messages

    def _get_history_field_name(self, signature: Type[Signature], inputs: dict[str, Any]) -> Optional[str]:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
            value = inputs.get(name)
            if hasattr(value, "messages") and isinstance(value.messages, list) and all(isinstance(m, dict) and "role" in m for m in value.messages):
                return name
            if hasattr(value, "_format"):
                try:
                    custom_format_output = value._format()
                    if isinstance(custom_format_output, list) and all(isinstance(m, dict) and "role" in m for m in custom_format_output):
                        return name
                except Exception:
                    pass
        return None

    def format_conversation_history(
        self,
        signature: Type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the conversation history.
            history_field_name: The name of the history field in the signature.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages.
        """
        history = inputs.get(history_field_name, None)

        if history is None:
            return []

        if hasattr(history, "_format"):
            del inputs[history_field_name]
            return history._format()

        conversation_history = getattr(history, "messages", None)

        if conversation_history is None:
            return []

        messages = []
        for message in conversation_history:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, message),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(signature, message),
                }
            )

        # Remove the history field from the inputs
        del inputs[history_field_name]

        return messages

    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            signature: The DSPy signature for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError
