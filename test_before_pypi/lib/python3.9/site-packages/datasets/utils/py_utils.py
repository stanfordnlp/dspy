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
"""Some python utils function and classes."""

import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse

import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm

from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
    Pickler,
    dump,
    dumps,
    pklregister,
)
from ._filelock import FileLock


try:  # pragma: no branch
    import typing_extensions as _typing_extensions
    from typing_extensions import Final, Literal
except ImportError:
    _typing_extensions = Literal = Final = None


logger = logging.get_logger(__name__)


# NOTE: When used on an instance method, the cache is shared across all
# instances and IS NOT per-instance.
# See
# https://stackoverflow.com/questions/14946264/python-lru-cache-decorator-per-instance
# For @property methods, use @memoized_property below.
memoize = functools.lru_cache


def size_str(size_in_bytes):
    """Returns a human readable size string.

    If size_in_bytes is None, then returns "Unknown size".

    For example `size_str(1.5 * datasets.units.GiB) == "1.50 GiB"`.

    Args:
        size_in_bytes: `int` or `None`, the size, in bytes, that we want to
            format as a human-readable size string.
    """
    if not size_in_bytes:
        return "Unknown size"

    _NAME_LIST = [("PiB", 2**50), ("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]

    size_in_bytes = float(size_in_bytes)
    for name, size_bytes in _NAME_LIST:
        value = size_in_bytes / size_bytes
        if value >= 1.0:
            return f"{value:.2f} {name}"
    return f"{int(size_in_bytes)} bytes"


def convert_file_size_to_int(size: Union[int, str]) -> int:
    """
    Converts a size expressed as a string with digits an unit (like `"50MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("PIB"):
        return int(size[:-3]) * (2**50)
    if size.upper().endswith("TIB"):
        return int(size[:-3]) * (2**40)
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("PB"):
        int_size = int(size[:-2]) * (10**15)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("TB"):
        int_size = int(size[:-2]) * (10**12)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(f"`size={size}` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def glob_pattern_to_regex(pattern):
    # partially taken from fsspec:
    # https://github.com/fsspec/filesystem_spec/blob/697d0f8133d8a5fbc3926e4761d7ecd51337ce50/fsspec/asyn.py#L735
    return (
        pattern.replace("\\", r"\\")
        .replace(".", r"\.")
        .replace("*", ".*")
        .replace("+", r"\+")
        .replace("//", "/")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("|", r"\|")
        .replace("^", r"\^")
        .replace("$", r"\$")
        .rstrip("/")
        .replace("?", ".")
    )


def string_to_dict(string: str, pattern: str) -> Dict[str, str]:
    """Un-format a string using a python f-string pattern.
    From https://stackoverflow.com/a/36838374

    Example::

        >>> p = 'hello, my name is {name} and I am a {age} year old {what}'
        >>> s = p.format(name='cody', age=18, what='quarterback')
        >>> s
        'hello, my name is cody and I am a 18 year old quarterback'
        >>> string_to_dict(s, p)
        {'age': '18', 'name': 'cody', 'what': 'quarterback'}

    Args:
        string (str): input string
        pattern (str): pattern formatted like a python f-string

    Returns:
        Dict[str, str]: dictionary of variable -> value, retrieved from the input using the pattern

    Raises:
        ValueError: if the string doesn't match the pattern
    """
    regex = re.sub(r"{(.+?)}", r"(?P<_\1>.+)", pattern)
    result = re.search(regex, string)
    if result is None:
        raise ValueError(f"String {string} doesn't match the pattern {pattern}")
    values = list(result.groups())
    keys = re.findall(r"{(.+?)}", pattern)
    _dict = dict(zip(keys, values))
    return _dict


def asdict(obj):
    """Convert an object to its dictionary representation recursively.

    <Added version="2.4.0"/>
    """

    # Implementation based on https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict

    def _is_dataclass_instance(obj):
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        return is_dataclass(obj) and not isinstance(obj, type)

    def _asdict_inner(obj):
        if _is_dataclass_instance(obj):
            result = {}
            for f in fields(obj):
                value = _asdict_inner(getattr(obj, f.name))
                if not f.init or value != f.default or f.metadata.get("include_in_asdict_even_if_is_default", False):
                    result[f.name] = value
            return result
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # obj is a namedtuple
            return type(obj)(*[_asdict_inner(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(_asdict_inner(v) for v in obj)
        elif isinstance(obj, dict):
            return {_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()}
        else:
            return copy.deepcopy(obj)

    if not isinstance(obj, dict) and not _is_dataclass_instance(obj):
        raise TypeError(f"{obj} is not a dict or a dataclass")

    return _asdict_inner(obj)


@contextmanager
def temporary_assignment(obj, attr, value):
    """Temporarily assign obj.attr to value."""
    original = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


@contextmanager
def temp_seed(seed: int, set_pytorch=False, set_tensorflow=False):
    """Temporarily set the random seed. This works for python numpy, pytorch and tensorflow."""
    np_state = np.random.get_state()
    np.random.seed(seed)

    if set_pytorch and config.TORCH_AVAILABLE:
        import torch

        torch_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)

        if torch.cuda.is_available():
            torch_cuda_states = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(seed)

    if set_tensorflow and config.TF_AVAILABLE:
        import tensorflow as tf
        from tensorflow.python.eager import context as tfpycontext

        tf_state = tf.random.get_global_generator()
        temp_gen = tf.random.Generator.from_seed(seed)
        tf.random.set_global_generator(temp_gen)

        if not tf.executing_eagerly():
            raise ValueError("Setting random seed for TensorFlow is only available in eager mode")

        tf_context = tfpycontext.context()  # eager mode context
        tf_seed = tf_context._seed
        tf_rng_initialized = hasattr(tf_context, "_rng")
        if tf_rng_initialized:
            tf_rng = tf_context._rng
        tf_context._set_global_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(np_state)

        if set_pytorch and config.TORCH_AVAILABLE:
            torch.random.set_rng_state(torch_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_states)

        if set_tensorflow and config.TF_AVAILABLE:
            tf.random.set_global_generator(tf_state)

            tf_context._seed = tf_seed
            if tf_rng_initialized:
                tf_context._rng = tf_rng
            else:
                delattr(tf_context, "_rng")


def unique_values(values):
    """Iterate over iterable and return only unique values in order."""
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value


def no_op_if_value_is_null(func):
    """If the value is None, return None, else call `func`."""

    def wrapper(value):
        return func(value) if value is not None else None

    return wrapper


def first_non_null_value(iterable):
    """Return the index and the value of the first non-null value in the iterable. If all values are None, return -1 as index."""
    for i, value in enumerate(iterable):
        if value is not None:
            return i, value
    return -1, None


def zip_dict(*dicts):
    """Iterate over items of dictionaries grouped by their keys."""
    for key in unique_values(itertools.chain(*dicts)):  # set merge all keys
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


class NonMutableDict(dict):
    """Dict where keys can only be added but not modified.

    Will raise an error if the user try to overwrite one key. The error message
    can be customized during construction. It will be formatted using {key} for
    the overwritten key.
    """

    def __init__(self, *args, **kwargs):
        self._error_msg = kwargs.pop(
            "error_msg",
            "Try to overwrite existing key: {key}",
        )
        if kwargs:
            raise ValueError("NonMutableDict cannot be initialized with kwargs.")
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(self._error_msg.format(key=key))
        return super().__setitem__(key, value)

    def update(self, other):
        if any(k in self for k in other):
            raise ValueError(self._error_msg.format(key=set(self) & set(other)))
        return super().update(other)


class classproperty(property):  # pylint: disable=invalid-name
    """Descriptor to be used as decorator for @classmethods."""

    def __get__(self, obj, objtype=None):
        return self.fget.__get__(None, objtype)()


def _single_map_nested(args):
    """Apply a function recursively to each element of a nested data struct."""
    function, data_struct, batched, batch_size, types, rank, disable_tqdm, desc = args

    # Singleton first to spare some computation
    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            return function([data_struct])[0]
        else:
            return function(data_struct)
    if (
        batched
        and not isinstance(data_struct, dict)
        and isinstance(data_struct, types)
        and all(not isinstance(v, (dict, types)) for v in data_struct)
    ):
        return [mapped_item for batch in iter_batched(data_struct, batch_size) for mapped_item in function(batch)]

    # Reduce logging to keep things readable in multiprocessing with tqdm
    if rank is not None and logging.get_verbosity() < logging.WARNING:
        logging.set_verbosity_warning()
    # Print at least one thing to fix tqdm in notebooks in multiprocessing
    # see https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
    if rank is not None and not disable_tqdm and any("notebook" in tqdm_cls.__name__ for tqdm_cls in tqdm.__mro__):
        print(" ", end="", flush=True)

    # Loop over single examples or batches and write to buffer/file if examples are to be updated
    pbar_iterable = data_struct.items() if isinstance(data_struct, dict) else data_struct
    pbar_desc = (desc + " " if desc is not None else "") + "#" + str(rank) if rank is not None else desc
    with hf_tqdm(pbar_iterable, disable=disable_tqdm, position=rank, unit="obj", desc=pbar_desc) as pbar:
        if isinstance(data_struct, dict):
            return {
                k: _single_map_nested((function, v, batched, batch_size, types, None, True, None)) for k, v in pbar
            }
        else:
            mapped = [_single_map_nested((function, v, batched, batch_size, types, None, True, None)) for v in pbar]
            if isinstance(data_struct, list):
                return mapped
            elif isinstance(data_struct, tuple):
                return tuple(mapped)
            else:
                return np.array(mapped)


def map_nested(
    function: Callable[[Any], Any],
    data_struct: Any,
    dict_only: bool = False,
    map_list: bool = True,
    map_tuple: bool = False,
    map_numpy: bool = False,
    num_proc: Optional[int] = None,
    parallel_min_length: int = 2,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    types: Optional[tuple] = None,
    disable_tqdm: bool = True,
    desc: Optional[str] = None,
) -> Any:
    """Apply a function recursively to each element of a nested data struct.

    Use multiprocessing if num_proc > 1 and the length of data_struct is greater than or equal to
    `parallel_min_length`.

    <Changed version="2.5.0">

    Before version 2.5.0, multiprocessing was not used if `num_proc` was greater than or equal to ``len(iterable)``.

    Now, if `num_proc` is greater than or equal to ``len(iterable)``, `num_proc` is set to ``len(iterable)`` and
    multiprocessing is used.

    </Changed>

    Args:
        function (`Callable`): Function to be applied to `data_struct`.
        data_struct (`Any`): Data structure to apply `function` to.
        dict_only (`bool`, default `False`): Whether only apply `function` recursively to `dict` values in
            `data_struct`.
        map_list (`bool`, default `True`): Whether also apply `function` recursively to `list` elements (besides `dict`
            values).
        map_tuple (`bool`, default `False`): Whether also apply `function` recursively to `tuple` elements (besides
            `dict` values).
        map_numpy (`bool, default `False`): Whether also apply `function` recursively to `numpy.array` elements (besides
            `dict` values).
        num_proc (`int`, *optional*): Number of processes.
            The level in the data struct used for multiprocessing is the first level that has smaller sub-structs,
            starting from the root.
        parallel_min_length (`int`, default `2`): Minimum length of `data_struct` required for parallel
            processing.
            <Added version="2.5.0"/>
        batched (`bool`, defaults to `False`):
            Provide batch of items to `function`.
            <Added version="2.19.0"/>
        batch_size (`int`, *optional*, defaults to `1000`):
            Number of items per batch provided to `function` if `batched=True`.
            If `batch_size <= 0` or `batch_size == None`, provide the full iterable as a single batch to `function`.
            <Added version="2.19.0"/>
        types (`tuple`, *optional*): Additional types (besides `dict` values) to apply `function` recursively to their
            elements.
        disable_tqdm (`bool`, default `True`): Whether to disable the tqdm progressbar.
        desc (`str`, *optional*): Prefix for the tqdm progressbar.

    Returns:
        `Any`
    """
    if types is None:
        types = []
        if not dict_only:
            if map_list:
                types.append(list)
            if map_tuple:
                types.append(tuple)
            if map_numpy:
                types.append(np.ndarray)
        types = tuple(types)

    # Singleton
    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            data_struct = [data_struct]
        mapped = function(data_struct)
        if batched:
            mapped = mapped[0]
        return mapped

    iterable = list(data_struct.values()) if isinstance(data_struct, dict) else data_struct

    if num_proc is None:
        num_proc = 1
    if any(isinstance(v, types) and len(v) > len(iterable) for v in iterable):
        mapped = [
            map_nested(
                function=function,
                data_struct=obj,
                num_proc=num_proc,
                parallel_min_length=parallel_min_length,
                batched=batched,
                batch_size=batch_size,
                types=types,
            )
            for obj in iterable
        ]
    elif num_proc != -1 and num_proc <= 1 or len(iterable) < parallel_min_length:
        if batched:
            if batch_size is None or batch_size <= 0:
                batch_size = max(len(iterable) // num_proc + int(len(iterable) % num_proc > 0), 1)
            iterable = list(iter_batched(iterable, batch_size))
        mapped = [
            _single_map_nested((function, obj, batched, batch_size, types, None, True, None))
            for obj in hf_tqdm(iterable, disable=disable_tqdm, desc=desc)
        ]
        if batched:
            mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".* is experimental and might be subject to breaking changes in the future\\.$",
                category=UserWarning,
            )
            if batched:
                if batch_size is None or batch_size <= 0:
                    batch_size = len(iterable) // num_proc + int(len(iterable) % num_proc > 0)
                iterable = list(iter_batched(iterable, batch_size))
            mapped = parallel_map(
                function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, _single_map_nested
            )
            if batched:
                mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]

    if isinstance(data_struct, dict):
        return dict(zip(data_struct.keys(), mapped))
    else:
        if isinstance(data_struct, list):
            return mapped
        elif isinstance(data_struct, tuple):
            return tuple(mapped)
        else:
            return np.array(mapped)


class NestedDataStructure:
    def __init__(self, data=None):
        self.data = data if data is not None else []

    def flatten(self, data=None):
        data = data if data is not None else self.data
        if isinstance(data, dict):
            return self.flatten(list(data.values()))
        elif isinstance(data, (list, tuple)):
            return [flattened for item in data for flattened in self.flatten(item)]
        else:
            return [data]


def has_sufficient_disk_space(needed_bytes, directory="."):
    try:
        free_bytes = disk_usage(os.path.abspath(directory)).free
    except OSError:
        return True
    return needed_bytes < free_bytes


def _convert_github_url(url_path: str) -> Tuple[str, Optional[str]]:
    """Convert a link to a file on a github repo in a link to the raw github object."""
    parsed = urlparse(url_path)
    sub_directory = None
    if parsed.scheme in ("http", "https", "s3") and parsed.netloc == "github.com":
        if "blob" in url_path:
            if not url_path.endswith(".py"):
                raise ValueError(f"External import from github at {url_path} should point to a file ending with '.py'")
            url_path = url_path.replace("blob", "raw")  # Point to the raw file
        else:
            # Parse github url to point to zip
            github_path = parsed.path[1:]
            repo_info, branch = github_path.split("/tree/") if "/tree/" in github_path else (github_path, "master")
            repo_owner, repo_name = repo_info.split("/")
            url_path = f"https://github.com/{repo_owner}/{repo_name}/archive/{branch}.zip"
            sub_directory = f"{repo_name}-{branch}"
    return url_path, sub_directory


def lock_importable_file(importable_local_file: str) -> FileLock:
    # Check the directory with a unique name in our dataset folder
    # path is: ./datasets/dataset_name/hash_from_code/script.py
    # we use a hash as subdirectory_name to be able to have multiple versions of a dataset/metric processing file together
    importable_directory_path = str(Path(importable_local_file).resolve().parent.parent)
    lock_path = importable_directory_path + ".lock"
    return FileLock(lock_path)


def get_imports(file_path: str) -> Tuple[str, str, str, str]:
    """Find whether we should import or clone additional files for a given processing script.
        And list the import.

    We allow:
    - library dependencies,
    - local dependencies and
    - external dependencies whose url is specified with a comment starting from "# From:' followed by the raw url to a file, an archive or a github repository.
        external dependencies will be downloaded (and extracted if needed in the dataset folder).
        We also add an `__init__.py` to each sub-folder of a downloaded folder so the user can import from them in the script.

    Note that only direct import in the dataset processing script will be handled
    We don't recursively explore the additional import to download further files.

    Example::

        import tensorflow
        import .c4_utils
        import .clicr.dataset-code.build_json_dataset  # From: https://raw.githubusercontent.com/clips/clicr/master/dataset-code/build_json_dataset
    """
    lines = []
    with open(file_path, encoding="utf-8") as f:
        lines.extend(f.readlines())

    logger.debug(f"Checking {file_path} for additional imports.")
    imports: List[Tuple[str, str, str, Optional[str]]] = []
    is_in_docstring = False
    for line in lines:
        docstr_start_match = re.findall(r'[\s\S]*?"""[\s\S]*?', line)

        if len(docstr_start_match) == 1:
            # flip True <=> False only if doctstring
            # starts at line without finishing
            is_in_docstring = not is_in_docstring

        if is_in_docstring:
            # import statements in doctstrings should
            # not be added as required dependencies
            continue

        match = re.match(r"^import\s+(\.?)([^\s\.]+)[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)", line, flags=re.MULTILINE)
        if match is None:
            match = re.match(
                r"^from\s+(\.?)([^\s\.]+)(?:[^\s]*)\s+import\s+[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)",
                line,
                flags=re.MULTILINE,
            )
            if match is None:
                continue
        if match.group(1):
            # The import starts with a '.', we will download the relevant file
            if any(imp[1] == match.group(2) for imp in imports):
                # We already have this import
                continue
            if match.group(3):
                # The import has a comment with 'From:', we'll retrieve it from the given url
                url_path = match.group(3)
                url_path, sub_directory = _convert_github_url(url_path)
                imports.append(("external", match.group(2), url_path, sub_directory))
            elif match.group(2):
                # The import should be at the same place as the file
                imports.append(("internal", match.group(2), match.group(2), None))
        else:
            if match.group(3):
                # The import has a comment with `From: git+https:...`, asks user to pip install from git.
                url_path = match.group(3)
                imports.append(("library", match.group(2), url_path, None))
            else:
                imports.append(("library", match.group(2), match.group(2), None))

    return imports


def copyfunc(func):
    result = types.FunctionType(func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__)
    result.__kwdefaults__ = func.__kwdefaults__
    return result


Y = TypeVar("Y")


def _write_generator_to_queue(queue: queue.Queue, func: Callable[..., Iterable[Y]], kwargs: dict) -> int:
    for i, result in enumerate(func(**kwargs)):
        queue.put(result)
    return i


def _get_pool_pid(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool]) -> Set[int]:
    return {f.pid for f in pool._pool}


def iflatmap_unordered(
    pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool],
    func: Callable[..., Iterable[Y]],
    *,
    kwargs_iterable: Iterable[dict],
) -> Iterable[Y]:
    initial_pool_pid = _get_pool_pid(pool)
    pool_changed = False
    manager_cls = Manager if isinstance(pool, multiprocessing.pool.Pool) else multiprocess.Manager
    with manager_cls() as manager:
        queue = manager.Queue()
        async_results = [
            pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_iterable
        ]
        try:
            while True:
                try:
                    yield queue.get(timeout=0.05)
                except Empty:
                    if all(async_result.ready() for async_result in async_results) and queue.empty():
                        break
                if _get_pool_pid(pool) != initial_pool_pid:
                    pool_changed = True
                    # One of the subprocesses has died. We should not wait forever.
                    raise RuntimeError(
                        "One of the subprocesses has abruptly died during map operation."
                        "To debug the error, disable multiprocessing."
                    )
        finally:
            if not pool_changed:
                # we get the result in case there's an error to raise
                [async_result.get(timeout=0.05) for async_result in async_results]


T = TypeVar("T")


def iter_batched(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    if n < 1:
        raise ValueError(f"Invalid batch size {n}")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
