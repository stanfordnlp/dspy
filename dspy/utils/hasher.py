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
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: bytes | list[bytes]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()
