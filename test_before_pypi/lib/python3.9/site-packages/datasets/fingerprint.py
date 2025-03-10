import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import xxhash

from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger


if TYPE_CHECKING:
    from .arrow_dataset import Dataset


logger = get_logger(__name__)


# Fingerprinting allows to have one deterministic fingerprint per dataset state.
# A dataset fingerprint is updated after each transform.
# Re-running the same transforms on a dataset in a different session results in the same fingerprint.
# This is possible thanks to a custom hashing function that works with most python objects.

# Fingerprinting is the main mechanism that enables caching.
# The caching mechanism allows to reload an existing cache file if it's already been computed.


#################
# Caching
#################

_CACHING_ENABLED = True
_TEMP_DIR_FOR_TEMP_CACHE_FILES: Optional["_TempCacheDir"] = None
_DATASETS_WITH_TABLE_IN_TEMP_DIR: Optional[weakref.WeakSet] = None


class _TempCacheDir:
    """
    A temporary directory for storing cached Arrow files with a cleanup that frees references to the Arrow files
    before deleting the directory itself to avoid permission errors on Windows.
    """

    def __init__(self):
        self.name = tempfile.mkdtemp(prefix=config.TEMP_CACHE_DIR_PREFIX)
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        for dset in get_datasets_with_cache_file_in_temp_dir():
            dset.__del__()
        if os.path.exists(self.name):
            try:
                shutil.rmtree(self.name)
            except Exception as e:
                raise OSError(
                    f"An error occured while trying to delete temporary cache directory {self.name}. Please delete it manually."
                ) from e

    def cleanup(self):
        if self._finalizer.detach():
            self._cleanup()


def maybe_register_dataset_for_temp_dir_deletion(dataset):
    """
    This function registers the datasets that have cache files in _TEMP_DIR_FOR_TEMP_CACHE_FILES in order
    to properly delete them before deleting the temporary directory.
    The temporary directory _TEMP_DIR_FOR_TEMP_CACHE_FILES is used when caching is disabled.
    """
    if _TEMP_DIR_FOR_TEMP_CACHE_FILES is None:
        return

    global _DATASETS_WITH_TABLE_IN_TEMP_DIR
    if _DATASETS_WITH_TABLE_IN_TEMP_DIR is None:
        _DATASETS_WITH_TABLE_IN_TEMP_DIR = weakref.WeakSet()
    if any(
        Path(_TEMP_DIR_FOR_TEMP_CACHE_FILES.name) in Path(cache_file["filename"]).parents
        for cache_file in dataset.cache_files
    ):
        _DATASETS_WITH_TABLE_IN_TEMP_DIR.add(dataset)


def get_datasets_with_cache_file_in_temp_dir():
    return list(_DATASETS_WITH_TABLE_IN_TEMP_DIR) if _DATASETS_WITH_TABLE_IN_TEMP_DIR is not None else []


def enable_caching():
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = True


def disable_caching():
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = False


@deprecated(
    "Use datasets.enable_caching() or datasets.disable_caching() instead. This function will be removed in a future version of datasets."
)
def set_caching_enabled(boolean: bool):
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use :func:`datasets.Dataset.save_to_disk` to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect :func:`datasets.load_dataset`. If you want to regenerate a dataset from scratch you should use
    the ``download_mode`` parameter in :func:`datasets.load_dataset`.
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = bool(boolean)


def is_caching_enabled() -> bool:
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`]] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    return bool(_CACHING_ENABLED)


def get_temporary_cache_files_directory() -> str:
    """Return a directory that is deleted when session closes."""
    global _TEMP_DIR_FOR_TEMP_CACHE_FILES
    if _TEMP_DIR_FOR_TEMP_CACHE_FILES is None:
        _TEMP_DIR_FOR_TEMP_CACHE_FILES = _TempCacheDir()
    return _TEMP_DIR_FOR_TEMP_CACHE_FILES.name


#################
# Hashing
#################


@deprecated("Use `copyreg.pickle` to register a custom reducer.")
def hashregister(*types):
    def proxy(func):
        for t in types:
            Hasher.dispatch[t] = func
        return func

    return proxy


class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: Dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    @deprecated("Use `Hasher.hash` instead.")
    def hash_default(cls, value: Any) -> str:
        return cls.hash(value)

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


#################
# Fingerprinting
#################

fingerprint_rng = random.Random()
# we show a warning only once when fingerprinting fails to avoid spam
fingerprint_warnings: Dict[str, bool] = {}


def generate_fingerprint(dataset: "Dataset") -> str:
    state = dataset.__dict__
    hasher = Hasher()
    for key in sorted(state):
        if key == "_fingerprint":
            continue
        hasher.update(key)
        hasher.update(state[key])
    # hash data files last modification timestamps as well
    for cache_file in dataset.cache_files:
        hasher.update(os.path.getmtime(cache_file["filename"]))
    return hasher.hexdigest()


def generate_random_fingerprint(nbits: int = 64) -> str:
    return f"{fingerprint_rng.getrandbits(nbits):0{nbits//4}x}"


def update_fingerprint(fingerprint, transform, transform_args):
    global fingerprint_warnings
    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:  # noqa various errors might raise here from pickle or dill
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                logger.warning(
                    f"Transform {transform} couldn't be hashed properly, a random hash was used instead. "
                    "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                    "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                    "This warning is only showed once. Subsequent hashing failures won't be showed."
                )
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            else:
                logger.info(f"Transform {transform} couldn't be hashed properly, a random hash was used instead.")
        else:
            logger.info(
                f"Transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
            )

        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:  # noqa various errors might raise here from pickle or dill
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    logger.warning(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. "
                        "Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. "
                        "If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. "
                        "This warning is only showed once. Subsequent hashing failures won't be showed."
                    )
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
                else:
                    logger.info(
                        f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead."
                    )
            else:
                logger.info(
                    f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled."
                )
            return generate_random_fingerprint()
    return hasher.hexdigest()


def validate_fingerprint(fingerprint: str, max_length=64):
    """
    Make sure the fingerprint is a non-empty string that is not longer that max_length=64 by default,
    so that the fingerprint can be used to name cache files without issues.
    """
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Invalid fingerprint '{fingerprint}': it should be a non-empty string.")
    for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
        if invalid_char in fingerprint:
            raise ValueError(
                f"Invalid fingerprint. Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{fingerprint}'. "
                f"They could create issues when creating cache files."
            )
    if len(fingerprint) > max_length:
        raise ValueError(
            f"Invalid fingerprint. Maximum lenth is {max_length} but '{fingerprint}' has length {len(fingerprint)}."
            "It could create issues when creating cache files."
        )


def format_transform_for_fingerprint(func: Callable, version: Optional[str] = None) -> str:
    """
    Format a transform to the format that will be used to update the fingerprint.
    """
    transform = f"{func.__module__}.{func.__qualname__}"
    if version is not None:
        transform += f"@{version}"
    return transform


def format_kwargs_for_fingerprint(
    func: Callable,
    args: Tuple,
    kwargs: Dict[str, Any],
    use_kwargs: Optional[List[str]] = None,
    ignore_kwargs: Optional[List[str]] = None,
    randomized_function: bool = False,
) -> Dict[str, Any]:
    """
    Format the kwargs of a transform to the format that will be used to update the fingerprint.
    """
    kwargs_for_fingerprint = kwargs.copy()
    if args:
        params = [p.name for p in inspect.signature(func).parameters.values() if p != p.VAR_KEYWORD]
        args = args[1:]  # assume the first argument is the dataset
        params = params[1:]
        kwargs_for_fingerprint.update(zip(params, args))
    else:
        del kwargs_for_fingerprint[
            next(iter(inspect.signature(func).parameters))
        ]  # assume the first key is the dataset

    # keep the right kwargs to be hashed to generate the fingerprint

    if use_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k in use_kwargs}
    if ignore_kwargs:
        kwargs_for_fingerprint = {k: v for k, v in kwargs_for_fingerprint.items() if k not in ignore_kwargs}
    if randomized_function:  # randomized functions have `seed` and `generator` parameters
        if kwargs_for_fingerprint.get("seed") is None and kwargs_for_fingerprint.get("generator") is None:
            _, seed, pos, *_ = np.random.get_state()
            seed = seed[pos] if pos < 624 else seed[0]
            kwargs_for_fingerprint["generator"] = np.random.default_rng(seed)

    # remove kwargs that are the default values

    default_values = {
        p.name: p.default for p in inspect.signature(func).parameters.values() if p.default != inspect._empty
    }
    for default_varname, default_value in default_values.items():
        if default_varname in kwargs_for_fingerprint and kwargs_for_fingerprint[default_varname] == default_value:
            kwargs_for_fingerprint.pop(default_varname)
    return kwargs_for_fingerprint


def fingerprint_transform(
    inplace: bool,
    use_kwargs: Optional[List[str]] = None,
    ignore_kwargs: Optional[List[str]] = None,
    fingerprint_names: Optional[List[str]] = None,
    randomized_function: bool = False,
    version: Optional[str] = None,
):
    """
    Wrapper for dataset transforms to update the dataset fingerprint using ``update_fingerprint``
    Args:
        inplace (:obj:`bool`):  If inplace is True, the fingerprint of the dataset is updated inplace.
            Otherwise, a parameter "new_fingerprint" is passed to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset.
        use_kwargs (:obj:`List[str]`, optional): optional white list of argument names to take into account
            to update the fingerprint to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset. By default all the arguments are used.
        ignore_kwargs (:obj:`List[str]`, optional): optional black list of argument names to take into account
            to update the fingerprint. Note that ignore_kwargs prevails on use_kwargs.
        fingerprint_names (:obj:`List[str]`, optional, defaults to ["new_fingerprint"]):
            If the dataset transforms is not inplace and returns a DatasetDict, then it can require
            several fingerprints (one per dataset in the DatasetDict). By specifying fingerprint_names,
            one fingerprint named after each element of fingerprint_names is going to be passed.
        randomized_function (:obj:`bool`, defaults to False): If the dataset transform is random and has
            optional parameters "seed" and "generator", then you can set randomized_function to True.
            This way, even if users set "seed" and "generator" to None, then the fingerprint is
            going to be randomly generated depending on numpy's current state. In this case, the
            generator is set to np.random.default_rng(np.random.get_state()[1][0]).
        version (:obj:`str`, optional): version of the transform. The version is taken into account when
            computing the fingerprint. If a datase transform changes (or at least if the output data
            that are cached changes), then one should increase the version. If the version stays the
            same, then old cached data could be reused that are not compatible with the new transform.
            It should be in the format "MAJOR.MINOR.PATCH".
    """

    if use_kwargs is not None and not isinstance(use_kwargs, list):
        raise ValueError(f"use_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if ignore_kwargs is not None and not isinstance(ignore_kwargs, list):
        raise ValueError(f"ignore_kwargs is supposed to be a list, not {type(use_kwargs)}")

    if inplace and fingerprint_names:
        raise ValueError("fingerprint_names are only used when inplace is False")

    fingerprint_names = fingerprint_names if fingerprint_names is not None else ["new_fingerprint"]

    def _fingerprint(func):
        if not inplace and not all(name in func.__code__.co_varnames for name in fingerprint_names):
            raise ValueError(f"function {func} is missing parameters {fingerprint_names} in signature")

        if randomized_function:  # randomized function have seed and generator parameters
            if "seed" not in func.__code__.co_varnames:
                raise ValueError(f"'seed' must be in {func}'s signature")
            if "generator" not in func.__code__.co_varnames:
                raise ValueError(f"'generator' must be in {func}'s signature")
        # this call has to be outside the wrapper or since __qualname__ changes in multiprocessing
        transform = format_transform_for_fingerprint(func, version=version)

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(
                func,
                args,
                kwargs,
                use_kwargs=use_kwargs,
                ignore_kwargs=ignore_kwargs,
                randomized_function=randomized_function,
            )

            if args:
                dataset: Dataset = args[0]
                args = args[1:]
            else:
                dataset: Dataset = kwargs.pop(next(iter(inspect.signature(func).parameters)))

            # compute new_fingerprint and add it to the args of not in-place transforms
            if inplace:
                new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
            else:
                for fingerprint_name in fingerprint_names:  # transforms like `train_test_split` have several hashes
                    if kwargs.get(fingerprint_name) is None:
                        kwargs_for_fingerprint["fingerprint_name"] = fingerprint_name
                        kwargs[fingerprint_name] = update_fingerprint(
                            dataset._fingerprint, transform, kwargs_for_fingerprint
                        )
                    else:
                        validate_fingerprint(kwargs[fingerprint_name])

            # Call actual function

            out = func(dataset, *args, **kwargs)

            # Update fingerprint of in-place transforms + update in-place history of transforms

            if inplace:  # update after calling func so that the fingerprint doesn't change if the function fails
                dataset._fingerprint = new_fingerprint

            return out

        wrapper._decorator_name_ = "fingerprint"
        return wrapper

    return _fingerprint
