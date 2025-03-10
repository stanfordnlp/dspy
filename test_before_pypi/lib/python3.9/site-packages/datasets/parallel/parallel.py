import contextlib
from multiprocessing import Pool, RLock

from tqdm.auto import tqdm

from ..utils import experimental, logging


logger = logging.get_logger(__name__)


class ParallelBackendConfig:
    backend_name = None


@experimental
def parallel_map(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func):
    """
    **Experimental.** Apply a function to iterable elements in parallel, where the implementation uses either
    multiprocessing.Pool or joblib for parallelization.

    Args:
        function (`Callable[[Any], Any]`): Function to be applied to `iterable`.
        iterable (`list`, `tuple` or `np.ndarray`): Iterable elements to apply function to.
        num_proc (`int`): Number of processes (if no backend specified) or jobs (using joblib).
        types (`tuple`): Additional types (besides `dict` values) to apply `function` recursively to their elements.
        disable_tqdm (`bool`): Whether to disable the tqdm progressbar.
        desc (`str`): Prefix for the tqdm progressbar.
        single_map_nested_func (`Callable`): Map function that applies `function` to an element from `iterable`.
            Takes a tuple of function, data_struct, types, rank, disable_tqdm, desc as input, where data_struct is an
            element of `iterable`, and `rank` is used for progress bar.
    """
    if ParallelBackendConfig.backend_name is None:
        return _map_with_multiprocessing_pool(
            function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
        )

    return _map_with_joblib(
        function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
    )


def _map_with_multiprocessing_pool(
    function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
):
    num_proc = num_proc if num_proc <= len(iterable) else len(iterable)
    split_kwds = []  # We organize the splits ourselve (contiguous splits)
    for index in range(num_proc):
        div = len(iterable) // num_proc
        mod = len(iterable) % num_proc
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        split_kwds.append((function, iterable[start:end], batched, batch_size, types, index, disable_tqdm, desc))

    if len(iterable) != sum(len(i[1]) for i in split_kwds):
        raise ValueError(
            f"Error dividing inputs iterable among processes. "
            f"Total number of objects {len(iterable)}, "
            f"length: {sum(len(i[1]) for i in split_kwds)}"
        )

    logger.info(
        f"Spawning {num_proc} processes for {len(iterable)} objects in slices of {[len(i[1]) for i in split_kwds]}"
    )
    initargs, initializer = None, None
    if not disable_tqdm:
        initargs, initializer = (RLock(),), tqdm.set_lock
    with Pool(num_proc, initargs=initargs, initializer=initializer) as pool:
        mapped = pool.map(single_map_nested_func, split_kwds)
    logger.info(f"Finished {num_proc} processes")
    mapped = [obj for proc_res in mapped for obj in proc_res]
    logger.info(f"Unpacked {len(mapped)} objects")

    return mapped


def _map_with_joblib(
    function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func
):
    # progress bar is not yet supported for _map_with_joblib, because tqdm couldn't accurately be applied to joblib,
    # and it requires monkey-patching joblib internal classes which is subject to change
    import joblib

    with joblib.parallel_backend(ParallelBackendConfig.backend_name, n_jobs=num_proc):
        return joblib.Parallel()(
            joblib.delayed(single_map_nested_func)((function, obj, batched, batch_size, types, None, True, None))
            for obj in iterable
        )


@experimental
@contextlib.contextmanager
def parallel_backend(backend_name: str):
    """
    **Experimental.**  Configures the parallel backend for parallelized dataset loading, which uses the parallelization
    implemented by joblib.

    Args:
        backend_name (str): Name of backend for parallelization implementation, has to be supported by joblib.

     Example usage:
     ```py
     with parallel_backend('spark'):
       dataset = load_dataset(..., num_proc=2)
     ```
    """
    ParallelBackendConfig.backend_name = backend_name

    if backend_name == "spark":
        from joblibspark import register_spark

        register_spark()

        # TODO: call create_cache_and_write_probe if "download" in steps
        # TODO: raise NotImplementedError when Dataset.map etc is called

    try:
        yield
    finally:
        ParallelBackendConfig.backend_name = None
