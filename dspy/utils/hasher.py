from pickle import dumps
from typing import Any

import xxhash

"""
The following class was pulled from the `datasets` package from Hugging Face.
The reason for vendoring this code is to avoid a hard dependency on `datasets`,
which is a large package that is not needed for the majority of use cases.

License: Apache License 2.0
Author: Hugging Face Inc.
URL: https://github.com/huggingface/datasets/blob/fa73ab472eecf9136a3daf7a0fbff16a3dffa7a6/src/datasets/fingerprint.py#L170
Changes: 2025-08-10 - Ran ruff to format the code to DSPy styles.
"""
class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: dict = {}

    def __init__(self):
        """Initialize an empty xxhash64 hasher state."""
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: bytes | list[bytes]) -> str:
        """Return a hex digest for one or more byte chunks.

        Args:
            value: A single bytes object or a list of bytes to hash in order.

        Returns:
            The xxhash64 hexadecimal digest.
        """
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        """Serialize and hash a Python object.

        Args:
            value: Any pickle-serializable Python object.

        Returns:
            The xxhash64 hexadecimal digest of the serialized object.
        """
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        """Update the running digest with a typed object payload.

        Args:
            value: Any pickle-serializable Python object to incorporate.
        """
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        """Return the hexadecimal digest of the current hasher state."""
        return self.m.hexdigest()
