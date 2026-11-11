"""Judgments: declared keys in, a distribution out (MAP-14).

changes/2026-09-17-judgments.md.  A judgment is a top-level property of a
``json_schema`` ``response_format`` that declares its answer set: a
boolean, a string ``enum``/``anyOf``-of-``const``, or an ordered integer
``enum``/``anyOf``-of-``const`` ``0..n-1`` — or, since
changes/2026-10-15-bounding-box.md, a **bounding box**: an object of the
four numbers ``top``, ``bottom``, ``left``, ``right`` (fractions of the
image's height and width).  A box declares no keys, so it is answered,
never scored.  This module reads that convention off a schema (§1),
rewrites judgment properties for the two wires that need it (§2), folds a
model's JSON text into a ``DataPart`` (§3), and offers the sugar that
EMITS the convention (``choice``, ``yes_no``, ``score``,
``bounding_box``, ``judgments``) so it need not be written by hand.
Nothing here touches the network.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .adaptation import adapt
from .errors import UnsupportedFeatureError
from .types import DataPart, JsonObject, JsonValue, Request, TextPart

JudgmentKind = Literal["boolean", "choice", "ordered", "box"]

MAX_ORDERED_LEVELS = 10  # Jev's Score ceiling (docs.typesafe.ai/primitives/score)
MAX_CHOICE_KEYS = 255  # Jev's Choice ceiling (docs.typesafe.ai/primitives/choice)

# changes/2026-10-15-bounding-box.md D1: a box is these four numbers, each a
# fraction of the image measured from its top-left corner — ``top`` and
# ``bottom`` of its height, ``left`` and ``right`` of its width (Jev's Box,
# docs.typesafe.ai/primitives/box).  The order is Jev's.
BOX_EDGES: tuple[str, ...] = ("top", "bottom", "left", "right")
BOX_EDGE_DESCRIPTIONS: dict[str, str] = {
    "top": "Top edge, as a fraction of the image's height from the top, from 0 to 1",
    "bottom": "Bottom edge, as a fraction of the image's height from the top, from 0 to 1",
    "left": "Left edge, as a fraction of the image's width from the left, from 0 to 1",
    "right": "Right edge, as a fraction of the image's width from the left, from 0 to 1",
}


@dataclass(frozen=True, slots=True)
class Judgment:
    """One declared judgment read off a schema property."""

    name: str
    kind: JudgmentKind
    keys: tuple[str, ...]
    instruction: str | None
    descriptions: dict[str, str | None]
    titles: dict[str, str | None]

    @property
    def ordered(self) -> bool:
        return self.kind == "ordered"

    @property
    def measurable(self) -> bool:
        """Whether a distribution over declared keys exists to measure.  A
        box declares no keys: it is answered, never scored (MAP-14 §1)."""
        return self.kind != "box"


# ─── §1 reading the convention ────────────────────────────────────────


def _const_branches(prop: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    branches = prop.get("anyOf")
    if not isinstance(branches, list) or not branches:
        return None
    if not all(isinstance(b, dict) and "const" in b for b in branches):
        return None
    return branches


def _is_box(prop: Mapping[str, Any]) -> bool:
    """An object whose properties are exactly the four edges, each a
    ``number`` (changes/2026-10-15-bounding-box.md D1).  Other keywords
    (descriptions, ``required``, ``additionalProperties``) are the caller's
    and travel verbatim."""
    if prop.get("type", "object") != "object":
        return False
    props = prop.get("properties")
    if not isinstance(props, dict) or set(props) != set(BOX_EDGES):
        return False
    return all(isinstance(edge, dict) and edge.get("type") == "number" for edge in props.values())


def _judgment_of(name: str, prop: Any) -> Judgment | None:
    if not isinstance(prop, dict):
        return None
    instruction = prop.get("description") if isinstance(prop.get("description"), str) and prop.get("description") else None
    if _is_box(prop):
        return Judgment(name, "box", (), instruction, {}, {})
    if prop.get("type") == "boolean":
        return Judgment(name, "boolean", ("true", "false"), instruction, {"true": None, "false": None}, {})
    enum = prop.get("enum")
    branches = _const_branches(prop)
    if isinstance(enum, list) and enum and branches is None:
        values = enum
        descs: dict[str, str | None] = {}
        titles: dict[str, str | None] = {}
    elif branches is not None and enum is None:
        values = [b["const"] for b in branches]
        descs = {str(b["const"]): (b.get("description") if isinstance(b.get("description"), str) else None) for b in branches}
        titles = {str(b["const"]): (b.get("title") if isinstance(b.get("title"), str) else None) for b in branches}
    else:
        return None
    if all(isinstance(v, str) and v for v in values):
        if prop.get("type") not in (None, "string"):
            return None
        keys = tuple(values)
        if len(set(keys)) != len(keys):
            return None
        return Judgment(name, "choice", keys, instruction, {k: descs.get(k) for k in keys}, {k: titles.get(k) for k in keys})
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        if prop.get("type") not in (None, "integer"):
            return None
        if list(values) != list(range(len(values))) or len(values) < 2:
            return None
        keys = tuple(str(v) for v in values)
        return Judgment(name, "ordered", keys, instruction, {k: descs.get(k) for k in keys}, {k: titles.get(k) for k in keys})
    return None


def judgments_in_schema(schema: Any) -> dict[str, Judgment]:
    """The judgments a json_schema declares, in property order (MAP-14 §1).

    Any property that is not one of the three shapes is ordinary
    structured output and is absent from the result.
    """
    if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}
    out: dict[str, Judgment] = {}
    for name, prop in props.items():
        if not isinstance(name, str):
            continue
        j = _judgment_of(name, prop)
        if j is not None:
            out[name] = j
    return out


def request_judgments(request: Request) -> dict[str, Judgment]:
    fmt = request.config.response_format
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return {}
    return judgments_in_schema(fmt.get("schema"))


def measurable_judgments(found: Mapping[str, Judgment]) -> dict[str, Judgment]:
    """The judgments with declared keys — the ones a distribution can be
    measured for.  Boxes are left out: they are answered, never scored."""
    return {name: j for name, j in found.items() if j.measurable}


def request_measurable_judgments(request: Request) -> dict[str, Judgment]:
    return measurable_judgments(request_judgments(request))


def non_judgment_properties(schema: Any, found: Mapping[str, Judgment]) -> tuple[str, ...]:
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return ()
    return tuple(name for name in props if name not in found)


# ─── §2 what a wire that measures nothing does with ``probabilities`` ──


def note_unmeasurable_probabilities(request: Request, provider: str) -> None:
    """MAP-14 §3 on a wire with no distribution: ``if_available`` records
    ``dropped``; ``required`` refuses before the wire (MAP-13 b).  A request
    whose judgments are all boxes asked for nothing measurable: nothing is
    recorded and nothing refused."""
    policy = request.config.probabilities
    if policy in (None, "off") or not request_measurable_judgments(request):
        return
    if policy == "required":
        raise UnsupportedFeatureError(
            f"{provider}: config.probabilities='required' but this wire cannot measure a distribution "
            "over the declared keys (it returns a pick only); use 'if_available' or a provider that can "
            "(typesafe, or a vLLM/SGLang server that honours logprob_token_ids)",
            provider=provider,
            feature="config.probabilities",
        )
    adapt(
        "config.probabilities",
        "dropped",
        "this wire cannot measure a distribution over the declared keys; the answer carries the pick only",
        asked=policy,
        provider=provider,
    )


def anthropic_schema(schema: JsonObject, found: Mapping[str, Judgment]) -> JsonObject:
    """A judgment property carrying both ``type`` and ``anyOf`` has its type
    moved into every branch: the Messages wire answers 400 "For 'anyOf',
    'type' is not supported" otherwise (receipted 2026-09-17).  Every other
    keyword stays verbatim (INV-050 exception)."""
    if not found:
        return schema
    out = copy.deepcopy(schema)
    for name in found:
        prop = out["properties"][name]
        branches = prop.get("anyOf")
        if "type" in prop and isinstance(branches, list):
            kind = prop.pop("type")
            for b in branches:
                b.setdefault("type", kind)
    return out


def gemini_schema(schema: JsonObject, found: Mapping[str, Judgment]) -> JsonObject:
    """Judgment properties go as ``enum`` with the per-key descriptions folded
    into the property description: ``responseJsonSchema`` ignores
    ``anyOf``/``const`` (it answered "Bordeaux-blend" for a declared set;
    receipted 2026-09-17) and honours ``enum``."""
    if not found:
        return schema
    out = copy.deepcopy(schema)
    for name, j in found.items():
        prop = out["properties"][name]
        if j.kind in ("boolean", "box") or "anyOf" not in prop:
            continue
        prop.pop("anyOf")
        prop["type"] = "integer" if j.ordered else "string"
        prop["enum"] = [int(k) for k in j.keys] if j.ordered else list(j.keys)
        lines = []
        for k in j.keys:
            label = j.titles.get(k)
            desc = j.descriptions.get(k)
            if label is None and desc is None:
                lines.append(k)  # a bare key still tells the model it is an option (receipted wire form)
                continue
            lines.append(f"{k} = " + (f"{label}: {desc}" if label and desc else (label or desc or "")))
        if any(j.titles.get(k) or j.descriptions.get(k) for k in j.keys):
            head = prop.get("description") or ""
            prop["description"] = (head + (" " if head else "") + ("Levels: " if j.ordered else "Options: ") + "; ".join(lines)).strip()
    return out


# ─── §3 the answer ────────────────────────────────────────────────────


def data_part_from_text(text: str, found: Mapping[str, Judgment]) -> DataPart | None:
    """The model's JSON object as a DataPart (value only), or None when the
    text is not a JSON object (a truncated answer stays a TextPart)."""
    if not found:
        return None
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return DataPart(value=value)


def replace_text_with_data(parts: Sequence[Any], found: Mapping[str, Judgment]) -> tuple[Any, ...]:
    """Swap the single text part of a judgment answer for its DataPart."""
    if not found:
        return tuple(parts)
    texts = [p for p in parts if isinstance(p, TextPart)]
    if len(texts) != 1:
        return tuple(parts)
    part = data_part_from_text(texts[0].text, found)
    if part is None:
        return tuple(parts)
    if texts[0].continuation:
        part = DataPart(value=part.value, continuation=texts[0].continuation)
    return tuple(part if p is texts[0] else p for p in parts)


def normalize_logprobs(scores: Mapping[str, float]) -> dict[str, float]:
    """Softmax over log-scores: one normalisation over the key set."""
    top = max(scores.values())
    weights = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def expected_level(distribution: Mapping[str, float]) -> float:
    return sum(float(p) * int(k) for k, p in distribution.items())


# ─── §4 sugar that emits the convention ───────────────────────────────


def choice(instruction: str, options: Mapping[str, str | None] | Sequence[str]) -> JsonObject:
    """A choice judgment property: ``{key: description-or-None}`` or a list of keys."""
    if isinstance(options, Mapping):
        items = list(options.items())
    else:
        items = [(k, None) for k in options]
    if not items:
        raise ValueError("choice needs at least one option")
    if any(not isinstance(k, str) or not k for k, _ in items):
        raise TypeError("choice option keys must be non-empty strings")
    if len({k for k, _ in items}) != len(items):
        raise ValueError("choice option keys must be unique")
    prop: JsonObject = {"type": "string", "description": instruction}
    if all(d is None for _, d in items):
        prop["enum"] = [k for k, _ in items]
    else:
        prop["anyOf"] = [({"const": k, "description": d} if d else {"const": k}) for k, d in items]
    return prop


def yes_no(instruction: str) -> JsonObject:
    return {"type": "boolean", "description": instruction}


def score(instruction: str, levels: Mapping[str, str] | Sequence[str]) -> JsonObject:
    """An ordered judgment: levels low → high; ``{name: description}`` or descriptions."""
    if isinstance(levels, Mapping):
        items = list(levels.items())
    else:
        items = [(None, d) for d in levels]
    if len(items) < 2:
        raise ValueError("score needs at least two levels")
    if len(items) > MAX_ORDERED_LEVELS:
        raise ValueError(f"score takes at most {MAX_ORDERED_LEVELS} levels")
    branches = []
    for i, (name, desc) in enumerate(items):
        b: JsonObject = {"const": i}
        if name:
            b["title"] = name
        if desc:
            b["description"] = desc
        branches.append(b)
    return {"type": "integer", "description": instruction, "anyOf": branches}


def bounding_box(instruction: str) -> JsonObject:
    """A bounding-box judgment: where in the request's image is what
    ``instruction`` describes (changes/2026-10-15-bounding-box.md).

    The answer is ``{"top", "bottom", "left", "right"}``, each a fraction
    from 0 to 1 of the image's height (``top``, ``bottom``) or width
    (``left``, ``right``), measured from the top-left corner; multiply by
    the image's size in pixels to draw or crop.  There is no "not found"
    answer: ask a ``yes_no`` beside it.  On ``typesafe`` this is Jev's Box
    question and needs Jev 1.14 (``model="jev-preview"`` or
    ``"jev-1.14.0"``); elsewhere it is ordinary structured output.

    The edges carry no ``minimum``/``maximum``: Anthropic refuses both on a
    number (HTTP 400, receipted 2026-10-15); the range is in each edge's
    description, which every wire passes to the model.
    """
    return {
        "type": "object",
        "description": instruction,
        "properties": {edge: {"type": "number", "description": BOX_EDGE_DESCRIPTIONS[edge]} for edge in BOX_EDGES},
        "required": list(BOX_EDGES),
        "additionalProperties": False,
    }


def judgments(*, name: str = "judgments", strict: bool = True, **properties: JsonObject) -> JsonObject:
    """A ``response_format`` declaring the given judgment properties."""
    if not properties:
        raise ValueError("judgments needs at least one property")
    schema: JsonObject = {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }
    return {"type": "json_schema", "name": name, "strict": strict, "schema": schema}


__all__ = [
    "Judgment", "BOX_EDGES", "judgments_in_schema", "request_judgments", "measurable_judgments",
    "request_measurable_judgments", "non_judgment_properties",
    "note_unmeasurable_probabilities", "anthropic_schema", "gemini_schema",
    "data_part_from_text", "replace_text_with_data", "normalize_logprobs", "expected_level",
    "choice", "yes_no", "score", "bounding_box", "judgments",
]
