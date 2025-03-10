import os
from typing import Dict, List, Tuple, TypeVar, Union


T = TypeVar("T")

ListLike = Union[List[T], Tuple[T, ...]]
NestedDataStructureLike = Union[T, List[T], Dict[str, T]]
PathLike = Union[str, bytes, os.PathLike]
