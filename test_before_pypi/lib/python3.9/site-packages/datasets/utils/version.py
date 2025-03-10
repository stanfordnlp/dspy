# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Version utils."""

import dataclasses
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Optional, Union


_VERSION_REG = re.compile(r"^(?P<major>\d+)" r"\.(?P<minor>\d+)" r"\.(?P<patch>\d+)$")


@total_ordering
@dataclass
class Version:
    """Dataset version `MAJOR.MINOR.PATCH`.

    Args:
        version_str (`str`):
            The dataset version.
        description (`str`):
            A description of what is new in this version.
        major (`str`):
        minor (`str`):
        patch (`str`):

    Example:

    ```py
    >>> VERSION = datasets.Version("1.0.0")
    ```
    """

    version_str: str
    description: Optional[str] = None
    major: Optional[Union[str, int]] = None
    minor: Optional[Union[str, int]] = None
    patch: Optional[Union[str, int]] = None

    def __post_init__(self):
        self.major, self.minor, self.patch = _str_to_version_tuple(self.version_str)

    def __repr__(self):
        return f"{self.tuple[0]}.{self.tuple[1]}.{self.tuple[2]}"

    @property
    def tuple(self):
        return self.major, self.minor, self.patch

    def _validate_operand(self, other):
        if isinstance(other, str):
            return Version(other)
        elif isinstance(other, Version):
            return other
        raise TypeError(f"{other} (type {type(other)}) cannot be compared to version.")

    def __eq__(self, other):
        try:
            other = self._validate_operand(other)
        except (TypeError, ValueError):
            return False
        else:
            return self.tuple == other.tuple

    def __lt__(self, other):
        other = self._validate_operand(other)
        return self.tuple < other.tuple

    def __hash__(self):
        return hash(_version_tuple_to_str(self.tuple))

    @classmethod
    def from_dict(cls, dic):
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dic.items() if k in field_names})

    def _to_yaml_string(self) -> str:
        return self.version_str


def _str_to_version_tuple(version_str):
    """Return the tuple (major, minor, patch) version extracted from the str."""
    res = _VERSION_REG.match(version_str)
    if not res:
        raise ValueError(f"Invalid version '{version_str}'. Format should be x.y.z with {{x,y,z}} being digits.")
    return tuple(int(v) for v in [res.group("major"), res.group("minor"), res.group("patch")])


def _version_tuple_to_str(version_tuple):
    """Return the str version from the version tuple (major, minor, patch)."""
    return ".".join(str(v) for v in version_tuple)
