from __future__ import annotations

from typing import Mapping
from typing import Sequence
from typing import Union


JSONSerializable = Union[
    Mapping[str, "JSONSerializable"],
    Sequence["JSONSerializable"],
    str,
    int,
    float,
    bool,
    None,
]

__all__ = ["JSONSerializable"]
