import re
from typing import Any

from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value, translate_field_type
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

_TAG_PATTERN = re.compile(r"<(?P<closing>/?)(?P<name>\w+)>")


def _scan_tags(text: str) -> list[tuple[int, int, bool, str]]:
    """Return every ``<name>``/``</name>`` tag as ``(start, end, is_closing, name)``."""
    return [
        (match.start(), match.end(), bool(match.group("closing")), match.group("name"))
        for match in _TAG_PATTERN.finditer(text)
    ]


def _balanced_partners(tags: list[tuple[int, int, bool, str]]) -> dict[int, int]:
    """Map each opening tag to the closing tag that balances it, per tag name.

    Computed with one stack per tag name in a single pass, so a value that legitimately
    contains a nested element of the same name (``<code><code>x</code></code>``) resolves
    to the *outer* closing tag rather than the first one.
    """
    partners: dict[int, int] = {}
    stacks: dict[str, list[int]] = {}
    for index, (_, _, is_closing, name) in enumerate(tags):
        if is_closing:
            stack = stacks.get(name)
            if stack:
                partners[stack.pop()] = index
        else:
            stacks.setdefault(name, []).append(index)
    return partners


def _next_closing_tags(tags: list[tuple[int, int, bool, str]]) -> list[int | None]:
    """For each tag, the index of the next closing tag sharing its name (or ``None``)."""
    following: list[int | None] = [None] * len(tags)
    latest: dict[str, int] = {}
    for index in range(len(tags) - 1, -1, -1):
        _, _, is_closing, name = tags[index]
        following[index] = latest.get(name)
        if is_closing:
            latest[name] = index
    return following


def _extract_field_blocks(completion: str, output_names: frozenset[str]) -> list[tuple[str, str, int, int]]:
    """Split ``completion`` into ``(field_name, raw_value, start, end)`` tuples, left to right.

    Runs in linear time. One rule goes beyond a plain lazy ``<name>(.*?)</name>`` scan, so that a
    value which is itself a same-named element is not truncated at the inner closing tag:

    * **Nesting.** When an opening tag is followed, after whitespace only, by another opening tag
      of the same name, the value is a nested document and the block ends at the depth-balanced
      closing tag. A tag name merely mentioned inside prose is not preceded by whitespace alone,
      so it keeps the lazy reading.

    The widened span has to be self-contained to be believable, so it is rejected unless every tag
    inside it pairs inside it, and unless it is free of other output fields. Without the first
    check the depth-balanced partner can be borrowed from a different element and the value gets
    stitched out of that element's markup; without the second, a later field is swallowed and its
    absence reported as a missing field. A rejected span falls back to the lazy reading.

    Everything else keeps the lazy reading too: a value that merely *ends* with its own closing tag
    is indistinguishable from a value followed by trailing commentary, so widening the block there
    would silently corrupt one reading to rescue the other. `_assert_unambiguous` reports those.
    """
    tags = _scan_tags(completion)
    partners = _balanced_partners(tags)
    next_closing = _next_closing_tags(tags)
    paired = dict(partners)
    paired.update({close: open_ for open_, close in partners.items()})

    def span_is_self_contained(open_index: int, close_index: int) -> bool:
        for inner in range(open_index + 1, close_index):
            partner = paired.get(inner)
            if partner is None or not open_index < partner < close_index:
                return False
            if not tags[inner][2] and tags[inner][3] in output_names and tags[inner][3] != tags[open_index][3]:
                return False
        return True

    blocks: list[tuple[str, str, int, int]] = []
    index = 0
    while index < len(tags):
        _, open_end, is_closing, name = tags[index]
        if is_closing:
            index += 1
            continue

        nested = (
            index + 1 < len(tags)
            and not tags[index + 1][2]
            and tags[index + 1][3] == name
            and not completion[open_end : tags[index + 1][0]].strip()
        )
        end = partners.get(index) if nested else None
        if end is not None and not span_is_self_contained(index, end):
            end = None
        if end is None:
            end = next_closing[index]
        if end is None:
            # No closing tag for this name anywhere later; treat the tag as plain text.
            index += 1
            continue

        blocks.append((name, completion[open_end : tags[end][0]], tags[index][0], tags[end][1]))
        index = end + 1
    return blocks


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
    def _assert_unambiguous(
        signature: type[Signature],
        completion: str,
        fields: dict[str, str],
        spans: dict[str, int],
        blocks: list[tuple[int, int]],
    ) -> None:
        """Raise if a field's value is cut short by a closing tag that may belong to the value.

        Nesting is decided by `_extract_field_blocks`, which is exact. What remains is the case
        no parser can settle: `<answer>a</answer> b</answer>` reads either as the value `a`
        followed by chatter or as the value `a</answer> b`, and the two are token-identical.
        Report it rather than silently picking one and returning the wrong string.

        `spans` holds where each output field's block ended; `blocks` holds every block found, in
        scan order, so they are sorted by start and never overlap. Blanking each block to spaces
        once, keeping every offset, leaves exactly the text no block accounts for: a closing tag
        surviving that mask is by construction in open text, and so is any text between two such
        copies. A tag can neither straddle a block boundary nor be forged out of blanks, because a
        block both starts and ends on the only `<` and `>` a tag has.
        """
        # Masking only ever blanks characters, so a tag absent from the raw text after a value is
        # absent from the mask too: the common well-formed completion needs no mask at all.
        if all(completion.find(f"</{name}>", block_end) == -1 for name, block_end in spans.items()):
            return

        pieces = []
        cursor = 0
        for start, end in blocks:
            pieces.append(completion[cursor:start])
            pieces.append(" " * (end - start))
            cursor = end
        pieces.append(completion[cursor:])
        masked = "".join(pieces)

        for name, block_end in spans.items():
            closing_tag = f"</{name}>"
            scanned = block_end
            while True:
                found = masked.find(closing_tag, scanned)
                if found == -1:
                    break

                # Whitespace alone between the value and a surplus copy means a duplicated-tag
                # slip: the readings differ only by a copy of the tag itself, so no text the model
                # wrote is lost and the parse stays silent. Keep scanning for a later copy.
                if not masked[scanned:found].strip():
                    scanned = found + len(closing_tag)
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
        """Extract each output field from its `<field_name>...</field_name>` block.

        A value holding a nested element of the same name is recovered in full; a value that
        merely ends with its own closing tag is ambiguous and raises `AdapterParseError` rather
        than being silently truncated. The first complete block wins for a repeated field.
        """
        fields = {}
        spans: dict[str, int] = {}
        blocks: list[tuple[int, int]] = []
        for name, content, start, end in _extract_field_blocks(completion, frozenset(signature.output_fields)):
            blocks.append((start, end))
            if name in signature.output_fields and name not in fields:
                fields[name] = content.strip()
                spans[name] = end
        # Report a missing field before reading a stray tag of an earlier field as an ambiguity.
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
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
