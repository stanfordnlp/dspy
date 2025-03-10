import enum


class StudyDirection(enum.IntEnum):
    """Direction of a :class:`~optuna.study.Study`.

    Attributes:
        NOT_SET:
            Direction has not been set.
        MINIMIZE:
            :class:`~optuna.study.Study` minimizes the objective function.
        MAXIMIZE:
            :class:`~optuna.study.Study` maximizes the objective function.
    """

    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2
