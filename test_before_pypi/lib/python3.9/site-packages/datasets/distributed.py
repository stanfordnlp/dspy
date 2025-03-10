from typing import TypeVar

from .arrow_dataset import Dataset, _split_by_node_map_style_dataset
from .iterable_dataset import IterableDataset, _split_by_node_iterable_dataset


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


def split_dataset_by_node(dataset: DatasetType, rank: int, world_size: int) -> DatasetType:
    """
    Split a dataset for the node at rank `rank` in a pool of nodes of size `world_size`.

    For map-style datasets:

    Each node is assigned a chunk of data, e.g. rank 0 is given the first chunk of the dataset.
    To maximize data loading throughput, chunks are made of contiguous data on disk if possible.

    For iterable datasets:

    If the dataset has a number of shards that is a factor of `world_size` (i.e. if `dataset.n_shards % world_size == 0`),
    then the shards are evenly assigned across the nodes, which is the most optimized.
    Otherwise, each node keeps 1 example out of `world_size`, skipping the other examples.

    Args:
        dataset ([`Dataset`] or [`IterableDataset`]):
            The dataset to split by node.
        rank (`int`):
            Rank of the current node.
        world_size (`int`):
            Total number of nodes.

    Returns:
        [`Dataset`] or [`IterableDataset`]: The dataset to be used on the node at rank `rank`.
    """
    if isinstance(dataset, Dataset):
        return _split_by_node_map_style_dataset(dataset, rank=rank, world_size=world_size)
    else:
        return _split_by_node_iterable_dataset(dataset, rank=rank, world_size=world_size)
