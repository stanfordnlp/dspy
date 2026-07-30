import re
from typing import Any

from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value, translate_field_type
from dspy.signatures.signature import Signature


class XMLAdapter(ChatAdapter):
    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            formatted = format_field_value(field_info=field.info, value=field_value)
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
    def _extract_field_values(signature: type[Signature], completion: str) -> dict[str, str]:
        """Extract the raw text of each output field, anchored to the signature's field names.

        The XML wire format does not escape values, so a value can legitimately contain its own
        closing tag, e.g. a `code` field whose body contains `</code>`. Scanning the completion
        with a single self-backreferencing pattern would end the value at the first closing tag
        and silently truncate it. Instead, walk the completion left to right and, for each field,
        take the last closing tag before the next expected field's opening tag: that boundary is
        the widest span that cannot belong to another field.
        """
        pending = list(signature.output_fields)
        opening_patterns = {name: re.compile(rf"<{re.escape(name)}>") for name in pending}
        fields: dict[str, str] = {}
        cursor = 0

        while pending and cursor < len(completion):
            # The next field to extract is whichever expected field opens first, so fields
            # emitted out of signature order are still handled.
            opening = None
            for name in pending:
                match = opening_patterns[name].search(completion, cursor)
                if match and (opening is None or match.start() < opening[0].start()):
                    opening = (match, name)
            if opening is None:
                break

            match, name = opening
            value_start = match.end()
            # The opening tag of any field still to be extracted bounds this value. A second
            # opening tag for the same field bounds it too, so a repeated block is not merged.
            boundary = len(completion)
            for other in pending:
                next_opening = opening_patterns[other].search(completion, value_start)
                if next_opening:
                    boundary = min(boundary, next_opening.start())

            closing_tag = f"</{name}>"
            candidates = []
            search_from = value_start
            while True:
                found = completion.find(closing_tag, search_from)
                if found == -1:
                    break
                candidates.append(found)
                search_from = found + 1
            if not candidates:
                # An opening tag with no closing tag anywhere: skip it and keep scanning.
                cursor = value_start
                continue

            # Prefer closing tags before the boundary; if the boundary tag turned out to be part
            # of this field's value, no candidate precedes it and the whole list is reconsidered.
            in_bounds = [pos for pos in candidates if pos < boundary]
            pool = in_bounds or candidates

            # A closing tag really ends the value only if nothing but another field follows it.
            # Trailing prose that merely mentions a closing tag, e.g. "...always close with
            # `</answer>`", must not be absorbed, so a candidate qualifies only when the rest of
            # the completion is blank or starts with a pending field's opening tag. Whitespace is
            # skipped by index rather than by slicing so a long completion is not recopied per
            # candidate.
            successors = [opening_patterns[other] for other in pending if other != name]
            closing = None
            for pos in reversed(pool):
                after = pos + len(closing_tag)
                while after < len(completion) and completion[after].isspace():
                    after += 1
                if after == len(completion) or any(p.match(completion, after) for p in successors):
                    closing = pos
                    break
            if closing is None:
                # No terminal candidate: the first closing tag is the safest guess, which matches
                # the non-greedy behavior this adapter had before.
                closing = pool[0]

            fields[name] = completion[value_start:closing].strip()
            pending.remove(name)
            cursor = closing + len(closing_tag)

        return fields

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = self._extract_field_values(signature, completion)
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
