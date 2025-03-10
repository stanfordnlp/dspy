from typing import List

import numpy as np


def _number_of_shards_in_gen_kwargs(gen_kwargs: dict) -> int:
    """Return the number of possible shards according to the input gen_kwargs"""
    # Having lists of different sizes makes sharding ambigious, raise an error in this case
    # until we decide how to define sharding without ambiguity for users
    lists_lengths = {key: len(value) for key, value in gen_kwargs.items() if isinstance(value, list)}
    if len(set(lists_lengths.values())) > 1:
        raise RuntimeError(
            (
                "Sharding is ambiguous for this dataset: "
                + "we found several data sources lists of different lengths, and we don't know over which list we should parallelize:\n"
                + "\n".join(f"\t- key {key} has length {length}" for key, length in lists_lengths.items())
                + "\nTo fix this, check the 'gen_kwargs' and make sure to use lists only for data sources, "
                + "and use tuples otherwise. In the end there should only be one single list, or several lists with the same length."
            )
        )
    max_length = max(lists_lengths.values(), default=0)
    return max(1, max_length)


def _distribute_shards(num_shards: int, max_num_jobs: int) -> List[range]:
    """
    Get the range of shard indices per job.
    If num_shards<max_num_jobs, then num_shards jobs are given a range of one shard.
    The shards indices order is preserved: e.g. all the first shards are given the first job.
    Moreover all the jobs are given approximately the same number of shards.

    Example:

    ```python
    >>> _distribute_shards(2, max_num_jobs=4)
    [range(0, 1), range(1, 2)]
    >>> _distribute_shards(10, max_num_jobs=3)
    [range(0, 4), range(4, 7), range(7, 10)]
    ```
    """
    shards_indices_per_group = []
    for group_idx in range(max_num_jobs):
        num_shards_to_add = num_shards // max_num_jobs + (group_idx < (num_shards % max_num_jobs))
        if num_shards_to_add == 0:
            break
        start = shards_indices_per_group[-1].stop if shards_indices_per_group else 0
        shard_indices = range(start, start + num_shards_to_add)
        shards_indices_per_group.append(shard_indices)
    return shards_indices_per_group


def _split_gen_kwargs(gen_kwargs: dict, max_num_jobs: int) -> List[dict]:
    """Split the gen_kwargs into `max_num_job` gen_kwargs"""
    # Having lists of different sizes makes sharding ambigious, raise an error in this case
    num_shards = _number_of_shards_in_gen_kwargs(gen_kwargs)
    if num_shards == 1:
        return [dict(gen_kwargs)]
    else:
        shard_indices_per_group = _distribute_shards(num_shards=num_shards, max_num_jobs=max_num_jobs)
        return [
            {
                key: [value[shard_idx] for shard_idx in shard_indices_per_group[group_idx]]
                if isinstance(value, list)
                else value
                for key, value in gen_kwargs.items()
            }
            for group_idx in range(len(shard_indices_per_group))
        ]


def _merge_gen_kwargs(gen_kwargs_list: List[dict]) -> dict:
    return {
        key: [value for gen_kwargs in gen_kwargs_list for value in gen_kwargs[key]]
        if isinstance(gen_kwargs_list[0][key], list)
        else gen_kwargs_list[0][key]
        for key in gen_kwargs_list[0]
    }


def _shuffle_gen_kwargs(rng: np.random.Generator, gen_kwargs: dict) -> dict:
    """Return a shuffled copy of the input gen_kwargs"""
    # We must shuffle all the lists, and lists of the same size must have the same shuffling.
    # This way entangled lists of (shard, shard_metadata) are still in the right order.

    # First, let's generate the shuffled indices per list size
    list_sizes = {len(value) for value in gen_kwargs.values() if isinstance(value, list)}
    indices_per_size = {}
    for size in list_sizes:
        indices_per_size[size] = list(range(size))
        rng.shuffle(indices_per_size[size])
    # Now let's copy the gen_kwargs and shuffle the lists based on their sizes
    shuffled_kwargs = dict(gen_kwargs)
    for key, value in shuffled_kwargs.items():
        if isinstance(value, list):
            shuffled_kwargs[key] = [value[i] for i in indices_per_size[len(value)]]
    return shuffled_kwargs
