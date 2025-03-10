from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from optuna import logging
from optuna.study._study_direction import StudyDirection


_logger = logging.get_logger(__name__)


class FrozenStudy:
    """Basic attributes of a :class:`~optuna.study.Study`.

    This class is private and not referenced by Optuna users.

    Attributes:
        study_name:
            Name of the :class:`~optuna.study.Study`.
        direction:
            :class:`~optuna.study.StudyDirection` of the :class:`~optuna.study.Study`.

            .. note::
                This attribute is only available during single-objective optimization.
        directions:
            A list of :class:`~optuna.study.StudyDirection` objects.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` set with
            :func:`optuna.study.Study.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` internally
            set by Optuna.

    """

    def __init__(
        self,
        study_name: str,
        direction: StudyDirection | None,
        user_attrs: dict[str, Any],
        system_attrs: dict[str, Any],
        study_id: int,
        *,
        directions: Sequence[StudyDirection] | None = None,
    ):
        self.study_name = study_name
        if direction is None and directions is None:
            raise ValueError("Specify one of `direction` and `directions`.")
        elif directions is not None:
            self._directions = list(directions)
        elif direction is not None:
            self._directions = [direction]
        else:
            raise ValueError("Specify only one of `direction` and `directions`.")
        self.user_attrs = user_attrs
        self.system_attrs = system_attrs
        self._study_id = study_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return self._study_id < other._study_id

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return self._study_id <= other._study_id

    @property
    def direction(self) -> StudyDirection:
        if len(self._directions) > 1:
            raise RuntimeError(
                "This attribute is not available during multi-objective optimization."
            )

        return self._directions[0]

    @property
    def directions(self) -> list[StudyDirection]:
        return self._directions
