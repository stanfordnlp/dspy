from __future__ import annotations

from typing import TYPE_CHECKING


try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


if TYPE_CHECKING:
    from typing import BinaryIO


class ArtifactStore(Protocol):
    """A protocol defining the interface for an artifact backend.

    The methods defined in this protocol are not supposed to be directly called by library users.

    An artifact backend is responsible for managing the storage and retrieval
    of artifact data. The backend should provide methods for opening, writing
    and removing artifacts.
    """

    def open_reader(self, artifact_id: str) -> BinaryIO:
        """Open the artifact identified by the artifact_id.

        This method should return a binary file-like object in read mode, similar to
        ``open(..., mode="rb")``. If the artifact does not exist, an
        :exc:`~optuna.artifacts.exceptions.ArtifactNotFound` exception
        should be raised.

        Args:
            artifact_id: The identifier of the artifact to open.

        Returns:
            BinaryIO: A binary file-like object that can be read from.
        """
        ...

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        """Save the content to the backend.

        Args:
            artifact_id: The identifier of the artifact to write to.
            content_body: The content to write to the artifact.
        """
        ...

    def remove(self, artifact_id: str) -> None:
        """Remove the artifact identified by the artifact_id.

        This method should delete the artifact from the backend. If the artifact does not
        exist, an :exc:`~optuna.artifacts.exceptions.ArtifactNotFound` exception
        may be raised.

        Args:
            artifact_id: The identifier of the artifact to remove.
        """
        ...
