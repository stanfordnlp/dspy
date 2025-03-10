from optuna.exceptions import OptunaError


class ArtifactNotFound(OptunaError):
    """Exception raised when an artifact is not found.

    It is typically raised while calling
    :meth:`~optuna.artifacts._protocol.ArtifactStore.open_reader` or
    :meth:`~optuna.artifacts._protocol.ArtifactStore.remove` methods.
    """

    ...
