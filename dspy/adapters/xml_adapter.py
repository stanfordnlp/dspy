import re
from typing import Any

from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value, translate_field_type
from dspy.signatures.signature import Signature


class XMLAdapter(ChatAdapter):
    field_pattern = re.compile(r"<(?P<name>\w+)>((?P<content>.*?))</\1>", re.DOTALL)

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            formatted = format_field_value(field_info=field.info, value=field_value)
            # A value holding its own closing tag would make the block ambiguous to read back,
            # so escape that one sequence; `parse` reverses it.
            formatted = self._escape_closing_tag(field.name, formatted)
            output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    def format_field_structure(self, signature: type[Signature]) -> str:
        """
        XMLAdapter requires input and output fields to be wrapped in XML tags like `<field_name>`.
        """

        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        return "\n\n".join(parts).strip()

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]

        messages.append(self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): inputs.get(k)
                for k, v in signature.input_fields.items() if k in inputs
            },
        ))

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        return self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        message = "Respond with the corresponding output fields wrapped in XML tags "
        message += ", then ".join(f"`<{f}>`" for f in signature.output_fields)
        message += "."
        return message

    @staticmethod
    def _escape_closing_tag(name: str, value: str) -> str:
        """Escape a field's own closing tag inside its value so the block stays unambiguous."""
        return value.replace(f"</{name}>", f"&lt;/{name}>")

    @staticmethod
    def _unescape_closing_tag(name: str, value: str) -> str:
        """Inverse of `_escape_closing_tag`."""
        return value.replace(f"&lt;/{name}>", f"</{name}>")

    @staticmethod
    def _assert_unambiguous(signature: type[Signature], completion: str, fields: dict[str, str]) -> None:
        """Raise if a field's block holds an unmatched closing tag that hides part of its value.

        The wire format does not escape values, so `<answer>a</answer> b</answer>` is genuinely
        ambiguous: `" b"` may be the rest of the value or trailing chatter. Nothing in the text
        distinguishes the two, so the adapter reports the ambiguity instead of silently keeping
        whichever reading the scan happened to reach.
        """
        for name in fields:
            opening_tag, closing_tag = f"<{name}>", f"</{name}>"
            # A repeated *block* pairs every closing tag with an opening one and is not ambiguous;
            # only a surplus closing tag can hide content.
            if completion.count(closing_tag) <= completion.count(opening_tag):
                continue

            opened = completion.find(opening_tag)
            chosen = completion.find(closing_tag, opened + len(opening_tag))
            last = completion.rfind(closing_tag)
            if chosen == -1 or last <= chosen:
                continue

            # Whitespace and further copies of the tag between the two are a duplicated-tag slip,
            # not lost content: the value is the same either way.
            between = completion[chosen + len(closing_tag) : last].replace(closing_tag, "")
            if not between.strip():
                continue

            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
                message=(
                    f"Field `{name}` contains an unmatched `{closing_tag}`, so its value cannot be "
                    f"determined: the text after the first `{closing_tag}` may belong to the value or "
                    f"may be trailing commentary. Parsing it either way risks silently returning the "
                    f"wrong value. Emit the field with `{closing_tag}` escaped as `&lt;/{name}>`, or "
                    f"use ChatAdapter or JSONAdapter, whose wire formats delimit values unambiguously."
                ),
            )

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            if name in signature.output_fields and name not in fields:
                fields[name] = content
        # A surplus closing tag makes the value ambiguous; fail loudly rather than truncate.
        self._assert_unambiguous(signature, completion, fields)
        fields = {k: self._unescape_closing_tag(k, v) for k, v in fields.items()}
        # Cast values using base class parse_value helper
        for k, v in fields.items():
            fields[k] = self._parse_field_value(signature.output_fields[k], v, completion, signature)
        if fields.keys() != signature.output_fields.keys():
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
        return fields

    def _parse_field_value(self, field_info, raw, completion, signature):
        from dspy.adapters.utils import parse_value

        try:
            return parse_value(raw, field_info.annotation)
        except Exception as e:
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse field {field_info} with value {raw}: {e}",
            )
