from dataclasses import dataclass
from typing import Literal

from dspy.utils.callback import OptimizerEvent

__all__ = [
    "CandidateProposed",
    "CandidateSelected",
    "OptimizerEvent",
    "RandomSearchCandidateProposed",
]


@dataclass(frozen=True, slots=True)
class CandidateProposed(OptimizerEvent):
    """A logical optimizer candidate that is about to be constructed.

    The index is proposal order within one optimizer callback call ID. It is
    not an index into a score-sorted result.
    """

    candidate_index: int


@dataclass(frozen=True, slots=True)
class RandomSearchCandidateProposed(CandidateProposed):
    """RandomSearch-native details for a proposed candidate."""

    seed: int
    kind: Literal["zero_shot", "labeled_few_shot", "unshuffled_bootstrap", "shuffled_bootstrap"]


@dataclass(frozen=True, slots=True)
class CandidateSelected(OptimizerEvent):
    """The candidate returned by an optimizer run.

    ``score`` uses DSPy Evaluate's aggregate percentage scale from 0 to 100.
    """

    candidate_index: int
    score: float
