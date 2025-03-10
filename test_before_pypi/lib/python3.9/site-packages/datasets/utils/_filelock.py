#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License
"""Utilities to handle file locking in `datasets`."""

import os

from filelock import FileLock as FileLock_
from filelock import UnixFileLock
from filelock import __version__ as _filelock_version
from packaging import version


class FileLock(FileLock_):
    """
    A `filelock.FileLock` initializer that handles long paths.
    It also uses the current umask for lock files.
    """

    MAX_FILENAME_LENGTH = 255

    def __init__(self, lock_file, *args, **kwargs):
        # The "mode" argument is required if we want to use the current umask in filelock >= 3.10
        # In previous previous it was already using the current umask.
        if "mode" not in kwargs and version.parse(_filelock_version) >= version.parse("3.10.0"):
            umask = os.umask(0o666)
            os.umask(umask)
            kwargs["mode"] = 0o666 & ~umask
        lock_file = self.hash_filename_if_too_long(lock_file)
        super().__init__(lock_file, *args, **kwargs)

    @classmethod
    def hash_filename_if_too_long(cls, path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        filename = os.path.basename(path)
        max_filename_length = cls.MAX_FILENAME_LENGTH
        if issubclass(cls, UnixFileLock):
            max_filename_length = min(max_filename_length, os.statvfs(os.path.dirname(path)).f_namemax)
        if len(filename) > max_filename_length:
            dirname = os.path.dirname(path)
            hashed_filename = str(hash(filename))
            new_filename = (
                filename[: max_filename_length - len(hashed_filename) - 8] + "..." + hashed_filename + ".lock"
            )
            return os.path.join(dirname, new_filename)
        else:
            return path
