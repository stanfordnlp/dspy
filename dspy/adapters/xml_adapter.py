import heapq
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable
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


def _first_straddling_close(tags: list[tuple[int, int, bool, str]], partners: dict[int, int]) -> list[int]:
    """For each tag, the earliest closing tag after it whose own opening tag lies before it.

    A closing tag with no opening tag at all straddles every position, so it is pending from the
    start. `len(tags)` means there is none, so every closing tag after that position pairs with an
    opening tag at or after it.
    """
    paired_closes = set(partners.values())
    pending = [index for index, (_, _, is_closing, _) in enumerate(tags) if is_closing and index not in paired_closes]
    heapq.heapify(pending)

    straddling: list[int] = [len(tags)] * len(tags)
    for index in range(len(tags)):
        while pending and pending[0] <= index:
            heapq.heappop(pending)
        straddling[index] = pending[0] if pending else len(tags)
        partner = partners.get(index)
        if partner is not None:
            heapq.heappush(pending, partner)
    return straddling


def _span_acceptor(
    tags: list[tuple[int, int, bool, str]],
    partners: dict[int, int],
    output_names: frozenset[str],
) -> Callable[[int, int], bool]:
    """Build the test deciding whether a depth-balanced span is believable as a single value.

    A span qualifies when every tag inside it pairs inside it, and when it holds no opening tag of
    a different output field. Walking a span to answer that costs its length, and a completion
    made of spans that are all rejected then costs their sum, so each question is instead answered
    from prefix data in constant time:

    * A tag inside the span pairs to the *left* of it only if some closing tag inside pairs with an
      opening tag before the span, which `_first_straddling_close` reports directly.
    * With that ruled out every closing tag inside pairs inside, so an opening tag that pairs to
      the *right* of the span, or with nothing at all, is exactly a surplus of opening tags over
      closing ones, which the running `balance` measures.
    * Opening tags naming an output field are counted the same way, discounting repeats of the
      span's own name, which are the nesting being recovered rather than a swallowed field.
    """
    straddling_close = _first_straddling_close(tags, partners)
    balance = [0] * (len(tags) + 1)
    field_opens = [0] * (len(tags) + 1)
    opens_by_name: dict[str, list[int]] = {}
    for index, (_, _, is_closing, name) in enumerate(tags):
        balance[index + 1] = balance[index] + (-1 if is_closing else 1)
        field_opens[index + 1] = field_opens[index] + (0 if is_closing or name not in output_names else 1)
        if not is_closing:
            opens_by_name.setdefault(name, []).append(index)

    def accepts(open_index: int, close_index: int) -> bool:
        if straddling_close[open_index] < close_index or balance[close_index] != balance[open_index + 1]:
            return False
        inside_field_opens = field_opens[close_index] - field_opens[open_index + 1]
        name = tags[open_index][3]
        if name in output_names:
            repeats = opens_by_name[name]
            inside_field_opens -= bisect_left(repeats, close_index) - bisect_right(repeats, open_index)
        return inside_field_opens == 0

    return accepts


def _extract_field_blocks(completion: str, output_names: frozenset[str]) -> list[tuple[str, int, int, int, int]]:
    """Split ``completion`` into ``(field_name, value_start, value_end, start, end)``, left to right.

    Blocks are reported as offsets into ``completion`` rather than as substrings: ``value_start``
    and ``value_end`` bound the raw value, ``start`` and ``end`` the whole block including its
    tags. Callers slice only the values they keep, so a completion full of tags naming nothing in
    the signature costs no copies.

    One rule goes beyond a plain lazy ``<name>(.*?)</name>`` scan, so that a value which is itself
    a same-named element is not truncated at the inner closing tag:

    * **Nesting.** When an opening tag is followed, after whitespace only, by another opening tag
      of the same name, the value is a nested document and the block ends at the depth-balanced
      closing tag. The whitespace test is what separates a nested document from a mention: a
      mention that follows other text (`<answer>x<answer> ...`) keeps the lazy reading, while one
      that opens the value (`<answer> <answer> ...`) is read as nesting and, if a balancing close
      exists, widens the value past it. A mention with no second closing tag is unaffected either
      way, since nothing balances it.

    The widened span has to be self-contained to be believable, so `_span_acceptor` rejects it
    unless every tag inside it pairs inside it, and unless it is free of other output fields.
    Without the first check the depth-balanced partner can be borrowed from a different element and
    the value gets stitched out of that element's markup; without the second, a later field is
    swallowed and its absence reported as a missing field. A rejected span falls back to the lazy
    reading, and costs no more than an accepted one to rule out.

    Everything else keeps the lazy reading too: a value that merely *ends* with its own closing tag
    is indistinguishable from a value followed by trailing commentary, so widening the block there
    would silently corrupt one reading to rescue the other.

    The lazy reading is a truncation whenever the value did own that closing tag, and
    `_assert_unambiguous` reports it only when the surplus tag survives the block mask with real
    text before it. Two shapes escape that and stay silently truncated: a value ending with its own
    closing tag, where only whitespace separates the two readings, and a rejected span where a later
    block -- which need not belong to a declared field, so an unrelated `<note>...</note>` counts --
    either covers the surplus tag or is all that separates it from the value. Representing either
    needs escaping, which this wire format does not have. This paragraph is the one statement of
    those limits in the code -- everything else here that mentions them points at it, and the
    adapters guide restates them for users.
    """
    tags = _scan_tags(completion)
    partners = _balanced_partners(tags)
    next_closing = _next_closing_tags(tags)
    span_is_self_contained = _span_acceptor(tags, partners, output_names)

    blocks: list[tuple[str, int, int, int, int]] = []
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

        blocks.append((name, open_end, tags[end][0], tags[index][0], tags[end][1]))
        index = end + 1
    return blocks


class XMLAdapter(ChatAdapter):
    """Adapter that wraps every field in `<field_name>...</field_name>` tags.

    Each input and output field is rendered as its own XML element, and the response is read back
    with a single scan over the tags, tolerant of surrounding whitespace. A value that is itself a
    same-named element (`<code><code>x</code></code>`) is recovered in full.

    The wire format has no escaping, so a value containing its own closing tag cannot always be
    told apart from a value followed by trailing commentary. Rather than guess, `parse` raises
    `AdapterParseError` when the surplus closing tag is left over in open text with real text
    before it; `_extract_field_blocks` names the shapes that slip past that check and truncate
    silently. Use `ChatAdapter` or `JSONAdapter` for content that can contain the closing tag.

    Under `dspy.streamify` a value of the nested shape is buffered and delivered as one chunk
    rather than token by token, because the tag that ends it is not known until it arrives, so a
    streamed value never disagrees with `parse`.
    """

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
    def _field_value_end(field_name: str, content: str, sibling_names: frozenset[str] = frozenset()) -> int | None:
        """Index in `content` where `<field_name>`'s value ends, or None if it has not ended yet.

        `content` is everything after the field's opening tag; `sibling_names` are the signature's
        other output field names.

        The boundary is decided by running `_extract_field_blocks` -- the same scan `parse` uses --
        over the reconstructed block, rather than by reimplementing its rule. An earlier version
        did reimplement it and drifted twice, widening spans `parse` rejects, so agreement is left
        to be structural instead of asserted. None means the block has not closed yet under that
        scan, which mid-stream means more content may still complete it.

        Tag 0 of the reconstructed string is the field's own opening tag, so the first block is the
        value's own whenever it starts at offset 0. When tag 0 closes nowhere it yields no block at
        all and every later block starts past the opening tag, which is the not-closed-yet case.
        """
        if not XMLAdapter._nested_decision_ready(field_name, content, sibling_names):
            return None

        opening = f"<{field_name}>"
        blocks = _extract_field_blocks(opening + content, sibling_names | {field_name})
        if not blocks:
            return None
        _, _, value_end, block_start, _ = blocks[0]
        if block_start != 0:
            return None
        return value_end - len(opening)

    @staticmethod
    def _nested_decision_ready(field_name: str, content: str, sibling_names: frozenset[str]) -> bool:
        """Whether enough of a nested value has arrived for its boundary to be decidable.

        `_extract_field_blocks` answers for a finished completion: when a nested span has no
        balancing tag it falls back to the lazy reading, which is right at parse time and premature
        mid-stream, where the balancing tag may simply not have arrived. Two things settle it: the
        span closing, or a sibling field opening, which rejects the widening for good. Until one of
        them happens the answer can still change, so the caller must keep waiting.
        """
        if not XMLAdapter._value_opens_nested(field_name, content):
            return True
        return XMLAdapter._nested_boundary_scan(field_name, content, sibling_names)[0]

    @staticmethod
    def _nested_boundary_scan(
        field_name: str,
        content: str,
        sibling_names: frozenset[str],
        depth: int = 0,
    ) -> tuple[bool, bool, str, int]:
        """Walk the tags in `content` for what settles a nested value's boundary, resumably.

        Returns `(ready, closed, unscanned, depth)`. `ready` is `_nested_decision_ready`'s answer
        -- the span closed, or a sibling opened -- and `closed` says a `</field_name>` went by,
        which is the only thing a block can end at. So `ready and closed` is exactly when
        `_field_value_end` can answer, and a caller that waits for both never runs the full scan
        on a value whose boundary has not arrived.

        `ready` and `closed` report only what this pass saw; both are monotone in `content`, so a
        caller resuming chunk by chunk keeps its own running `or`, carries `depth` forward, and
        passes the next chunk with `unscanned` prefixed. Resuming that way is what keeps a streamed
        value linear in both work and copying: re-reading the buffer whenever a tag lands costs a
        pass over it per nested child, and rebuilding the buffer to re-read costs a copy of it per
        chunk.

        `unscanned` is the tail that may hold the front of a tag split across two chunks. It never
        reaches back over a tag already counted, and is otherwise one character short of the widest
        closing tag, which carries whole every field and sibling tag -- the only names this scan
        reacts to. A longer tag of some other name can be cut instead, which changes no answer,
        since the scan ignores it either way and resuming inside it starts past the only `<` it has.
        """
        ready = False
        closed = False
        last_end = 0
        for match in _TAG_PATTERN.finditer(content):
            last_end = match.end()
            name = match.group("name")
            is_closing = bool(match.group("closing"))
            if name == field_name:
                if not is_closing:
                    depth += 1
                else:
                    closed = True
                    if depth == 0:
                        ready = True
                    else:
                        depth -= 1
            elif not is_closing and name in sibling_names:
                ready = True

        widest_tag = max(len(name) for name in sibling_names | {field_name}) + 3
        return ready, closed, content[max(last_end, len(content) - widest_tag + 1) :], depth

    @staticmethod
    def _value_opens_nested(field_name: str, content: str) -> bool | None:
        """Whether a value is a nested same-named document. None while still undecidable.

        Decided from the first non-whitespace run after the opening tag, so a streamed value can
        be classified before it has fully arrived. None means the text so far is whitespace or a
        prefix of the opening tag and more is needed.
        """
        opening = f"<{field_name}>"
        stripped = content.lstrip()
        if stripped.startswith(opening):
            return True
        if not stripped or opening.startswith(stripped):
            return None
        return False

    @staticmethod
    def _assert_unambiguous(
        signature: type[Signature],
        completion: str,
        fields: dict[str, str],
        spans: dict[str, int],
        blocks: list[tuple[int, int]],
    ) -> None:
        """Raise if a field's value is cut short by a closing tag left over in open text.

        Nesting is decided by `_extract_field_blocks`. What remains is the case no parser can
        settle: `<answer>a</answer> b</answer>` reads either as the value `a` followed by chatter
        or as the value `a</answer> b`, and the two are token-identical. Report it rather than
        silently picking one and returning the wrong string. A truncation whose surplus tag is
        *not* left over in open text is invisible here; `_extract_field_blocks` lists those shapes.

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

        A value holding a nested element of the same name is recovered in full; a value whose own
        closing tag is left over in open text is ambiguous and raises `AdapterParseError` rather
        than being silently truncated. The first complete block wins for a repeated field.

        Two shapes stay silently truncated, because representing either needs escaping this wire
        format does not have; `_extract_field_blocks` names them and spells out why each is
        undecidable.
        """
        fields = {}
        spans: dict[str, int] = {}
        blocks: list[tuple[int, int]] = []
        for name, value_start, value_end, start, end in _extract_field_blocks(
            completion, frozenset(signature.output_fields)
        ):
            blocks.append((start, end))
            if name in signature.output_fields and name not in fields:
                fields[name] = completion[value_start:value_end].strip()
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
