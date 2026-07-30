import re
from typing import Any

from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value, translate_field_type
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError


class XMLAdapter(ChatAdapter):
    field_pattern = re.compile(r"<(?P<name>\w+)>((?P<content>.*?))</\1>", re.DOTALL)

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
    def _assert_unambiguous(
        signature: type[Signature],
        completion: str,
        fields: dict[str, str],
        spans: dict[str, tuple[int, int]],
        blocks: list[tuple[int, int]],
    ) -> None:
        """Raise if a field's value is cut short by a closing tag that may belong to the value.

        The wire format does not escape values, so `<answer>a</answer> b</answer>` is genuinely
        ambiguous: `" b"` may be the rest of the value or trailing chatter, and nothing in the
        text tells the two apart. Rather than silently keeping whichever reading the scan
        reached first, report it and let the caller pick a format that can express the value.

        `spans` holds each output field's own block; `blocks` holds every balanced block found,
        in scan order, so they are sorted by start and never overlap. Every cursor below only
        moves forward over them, which keeps a field's whole scan linear in the length of the
        completion however many surplus tags it holds.
        """
        for name, (_, block_end) in spans.items():
            closing_tag = f"</{name}>"
            opening_tag = f"<{name}>"

            containing = 0
            skipping = 0
            openings = 0
            closings = 0
            next_opening = completion.find(opening_tag, block_end)
            scanned = block_end
            hides_content = False

            search_from = block_end
            while True:
                found = completion.find(closing_tag, search_from)
                if found == -1:
                    break
                search_from = found + len(closing_tag)

                while next_opening != -1 and next_opening + len(opening_tag) <= found:
                    openings += 1
                    next_opening = completion.find(opening_tag, next_opening + len(opening_tag))
                while containing < len(blocks) and blocks[containing][1] <= found:
                    containing += 1
                preceding_closings = closings
                closings += 1

                # A later copy of this tag only matters if it sits in open text. Two things
                # account for one: nesting inside any balanced block, whether or not that block
                # is an output field, where the tag belongs to that block's value, and a repeated
                # block of this same field, where an unmatched opening tag ahead of it means it
                # closes that block rather than this one.
                if containing < len(blocks) and blocks[containing][0] <= found:
                    continue
                if openings > preceding_closings:
                    continue

                # Only text that no block accounts for can be hidden by the longer reading, so
                # classify the gap a piece at a time, skipping whole blocks and further copies of
                # the tag. Whitespace and bare copies in between mean a duplicated-tag slip: the
                # two readings still differ - "a" versus "a</answer>\n" - but the trailing copy
                # carries no content, so the shorter reading loses nothing and stays silent. Keep
                # scanning past it: a later copy may still hide content.
                while scanned < found and not hides_content:
                    while skipping < len(blocks) and blocks[skipping][1] <= scanned:
                        skipping += 1
                    if skipping < len(blocks) and blocks[skipping][0] <= scanned:
                        scanned = min(blocks[skipping][1], found)
                        continue
                    stop = min(blocks[skipping][0], found) if skipping < len(blocks) else found
                    hides_content = bool(completion[scanned:stop].replace(closing_tag, "").strip())
                    scanned = stop
                if not hides_content:
                    continue

                raise AdapterParseError(
                    adapter_name="XMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=fields,
                    message=(
                        f"Field `{name}` is followed by an unmatched `{closing_tag}`, so its value cannot "
                        f"be determined: the text after the first `{closing_tag}` may belong to the value "
                        f"or may be trailing commentary. Returning either reading risks silently giving "
                        f"back the wrong value. Use ChatAdapter or JSONAdapter, whose wire formats "
                        f"delimit values unambiguously, for content that can contain `{closing_tag}`."
                    ),
                )

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        spans: dict[str, tuple[int, int]] = {}
        blocks: list[tuple[int, int]] = []
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            blocks.append(match.span())
            if name in signature.output_fields and name not in fields:
                fields[name] = content
                spans[name] = match.span()
        # A field with no closing tag anywhere leaves a stray copy of an earlier field's tag in
        # open text, so report the missing field before reading that stray tag as an ambiguity.
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
        # A surplus closing tag makes the value ambiguous; fail loudly rather than truncate.
        self._assert_unambiguous(signature, completion, fields, spans, blocks)
        # Cast values using base class parse_value helper
        for k, v in fields.items():
            fields[k] = self._parse_field_value(signature.output_fields[k], v, completion, signature)
        return fields

    def _parse_field_value(self, field_info, raw, completion, signature):
        from dspy.adapters.utils import parse_value

        try:
            return parse_value(raw, field_info.annotation)
        except Exception as e:
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse field {field_info} with value {raw}: {e}",
            )
