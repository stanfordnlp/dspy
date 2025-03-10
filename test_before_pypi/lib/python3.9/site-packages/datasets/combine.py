from typing import List, Optional, TypeVar

from .arrow_dataset import Dataset, _concatenate_map_style_datasets, _interleave_map_style_datasets
from .dataset_dict import DatasetDict, IterableDatasetDict
from .info import DatasetInfo
from .iterable_dataset import IterableDataset, _concatenate_iterable_datasets, _interleave_iterable_datasets
from .splits import NamedSplit
from .utils import logging
from .utils.py_utils import Literal


logger = logging.get_logger(__name__)


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


def interleave_datasets(
    datasets: List[DatasetType],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
) -> DatasetType:
    """
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.

    You can use this function on a list of [`Dataset`] objects, or on a list of [`IterableDataset`] objects.

        - If `probabilities` is `None` (default) the new dataset is constructed by cycling between each source to get the examples.
        - If `probabilities` is not `None`, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    The resulting dataset ends when one of the source datasets runs out of examples except when `oversampling` is `True`,
    in which case, the resulting dataset ends when all datasets have ran out of examples at least one time.

    Note for iterable datasets:

    In a distributed setup or in PyTorch DataLoader workers, the stopping strategy is applied per process.
    Therefore the "first_exhausted" strategy on an sharded iterable dataset can generate less samples in total (up to 1 missing sample per subdataset per worker).

    Args:
        datasets (`List[Dataset]` or `List[IterableDataset]`):
            List of datasets to interleave.
        probabilities (`List[float]`, *optional*, defaults to `None`):
            If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        seed (`int`, *optional*, defaults to `None`):
            The random seed used to choose a source for each example.
        info ([`DatasetInfo`], *optional*):
            Dataset information, like description, citation, etc.
            <Added version="2.4.0"/>
        split ([`NamedSplit`], *optional*):
            Name of the dataset split.
            <Added version="2.4.0"/>
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now, `first_exhausted` and `all_exhausted`.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have `max_length_datasets*nb_dataset` samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
    Returns:
        [`Dataset`] or [`IterableDataset`]: Return type depends on the input `datasets`
        parameter. `Dataset` if the input is a list of `Dataset`, `IterableDataset` if the input is a list of
        `IterableDataset`.

    Example:

        For regular datasets (map-style):

        ```python
        >>> from datasets import Dataset, interleave_datasets
        >>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
        >>> d2 = Dataset.from_dict({"a": [10, 11, 12]})
        >>> d3 = Dataset.from_dict({"a": [20, 21, 22]})
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [10, 0, 11, 1, 2, 20, 12, 10, 0, 1, 2, 21, 0, 11, 1, 2, 0, 1, 12, 2, 10, 0, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42)
        >>> dataset["a"]
        [10, 0, 11, 1, 2]
        >>> dataset = interleave_datasets([d1, d2, d3])
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
        >>> d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
        >>> d3 = Dataset.from_dict({"a": [20, 21, 22, 23, 24]})
        >>> dataset = interleave_datasets([d1, d2, d3])
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22, 0, 13, 23, 1, 10, 24]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42)
        >>> dataset["a"]
        [10, 0, 11, 1, 2]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [10, 0, 11, 1, 2, 20, 12, 13, ..., 0, 1, 2, 0, 24]
        For datasets in streaming mode (iterable):

        >>> from datasets import load_dataset, interleave_datasets
        >>> d1 = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)
        >>> d2 = load_dataset("oscar", "unshuffled_deduplicated_fr", split="train", streaming=True)
        >>> dataset = interleave_datasets([d1, d2])
        >>> iterator = iter(dataset)
        >>> next(iterator)
        {'text': 'Mtendere Village was inspired by the vision...}
        >>> next(iterator)
        {'text': "Média de débat d'idées, de culture...}
        ```
    """
    from .arrow_dataset import Dataset
    from .iterable_dataset import IterableDataset

    if not datasets:
        raise ValueError("Unable to interleave an empty list of datasets.")
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError(f"{stopping_strategy} is not supported. Please enter a valid stopping_strategy.")
    if dataset_type is Dataset:
        return _interleave_map_style_datasets(
            datasets, probabilities, seed, info=info, split=split, stopping_strategy=stopping_strategy
        )
    else:
        return _interleave_iterable_datasets(
            datasets, probabilities, seed, info=info, split=split, stopping_strategy=stopping_strategy
        )


def concatenate_datasets(
    dsets: List[DatasetType],
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    axis: int = 0,
) -> DatasetType:
    """
    Converts a list of [`Dataset`] with the same schema into a single [`Dataset`].

    Args:
        dsets (`List[datasets.Dataset]`):
            List of Datasets to concatenate.
        info (`DatasetInfo`, *optional*):
            Dataset information, like description, citation, etc.
        split (`NamedSplit`, *optional*):
            Name of the dataset split.
        axis (`{0, 1}`, defaults to `0`):
            Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
            (horizontally).

            <Added version="1.6.0"/>

    Example:

    ```py
    >>> ds3 = concatenate_datasets([ds1, ds2])
    ```
    """

    if not dsets:
        raise ValueError("Unable to concatenate an empty list of datasets.")
    for i, dataset in enumerate(dsets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if dataset_type is Dataset:
        return _concatenate_map_style_datasets(dsets, info=info, split=split, axis=axis)
    else:
        return _concatenate_iterable_datasets(dsets, info=info, split=split, axis=axis)
