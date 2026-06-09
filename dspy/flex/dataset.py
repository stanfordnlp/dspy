from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy
from dspy.flex.exploration import FLEX_DIRNAME, _atomic_write_json

if TYPE_CHECKING:
    from dspy.primitives.example import Example

logger = logging.getLogger(__name__)

DATASET_FILENAME = "dataset.json"


def _serialize_example(ex: Example) -> dict[str, Any]:
    """Serialize one ``dspy.Example`` to a JSON-friendly record.

    ``toDict()`` captures the field store but not the input/label split, so the
    input keys are recorded separately. ``None`` (``with_inputs`` never called)
    is preserved distinctly from an empty set so the round-trip is exact.
    """
    input_keys = None if ex._input_keys is None else sorted(ex._input_keys)
    return {"data": ex.toDict(), "input_keys": input_keys}


def _deserialize_example(rec: dict[str, Any]) -> Example:
    ex = dspy.Example(**rec.get("data", {}))
    input_keys = rec.get("input_keys")
    if input_keys is not None:
        ex = ex.with_inputs(*input_keys)
    return ex


class DatasetStore:
    """Persists the train/val examples a ``FlexGEPA`` run optimized against.

    Lives at ``<root>/.flex/<flex_id>/dataset.json`` next to the rest of a Flex
    module's bookkeeping. Saving the dataset is what lets a later run re-optimize
    a hand-edited module via :meth:`dspy.Flex.improve` without the user having to
    re-supply the examples.

    When ``root`` is ``None`` (an in-memory Flex with no ``persist_to``), ``save``
    is a no-op and ``load`` returns ``None`` — callers don't have to null-check.
    """

    def __init__(self, root: str | Path | None, flex_id: str):
        self.flex_id = flex_id
        self.path: Path | None = (
            Path(root) / FLEX_DIRNAME / flex_id / DATASET_FILENAME if root is not None else None
        )

    def save(self, trainset: list[Example], valset: list[Example] | None = None) -> None:
        if self.path is None:
            return
        payload = {
            "flex_id": self.flex_id,
            "trainset": [_serialize_example(ex) for ex in trainset],
            # A valset that is the same object as the trainset (FlexGEPA's default
            # when no separate valset is passed) is stored as null to avoid
            # duplicating it on disk; load() reuses the trainset in that case.
            "valset": None
            if valset is None or valset is trainset
            else [_serialize_example(ex) for ex in valset],
        }
        _atomic_write_json(self.path, payload)
        logger.info("dspy.Flex %r: saved optimization dataset to %s", self.flex_id, self.path)

    def load(self) -> tuple[list[Example], list[Example]] | None:
        """Return ``(trainset, valset)`` or ``None`` when no dataset is stored."""
        if self.path is None or not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        trainset = [_deserialize_example(rec) for rec in data.get("trainset", [])]
        raw_val = data.get("valset")
        valset = trainset if raw_val is None else [_deserialize_example(rec) for rec in raw_val]
        return trainset, valset
