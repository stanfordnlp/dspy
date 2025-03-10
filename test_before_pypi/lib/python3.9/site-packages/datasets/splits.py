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
"""Splits related API."""

import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict


@dataclass
class SplitInfo:
    name: str = dataclasses.field(default="", metadata={"include_in_asdict_even_if_is_default": True})
    num_bytes: int = dataclasses.field(default=0, metadata={"include_in_asdict_even_if_is_default": True})
    num_examples: int = dataclasses.field(default=0, metadata={"include_in_asdict_even_if_is_default": True})
    shard_lengths: Optional[List[int]] = None

    # Deprecated
    # For backward compatibility, this field needs to always be included in files like
    # dataset_infos.json and dataset_info.json files
    # To do so, we always include it in the output of datasets.utils.py_utils.asdict(split_info)
    dataset_name: Optional[str] = dataclasses.field(
        default=None, metadata={"include_in_asdict_even_if_is_default": True}
    )

    @property
    def file_instructions(self):
        """Returns the list of dict(filename, take, skip)."""
        # `self.dataset_name` is assigned in `SplitDict.add()`.
        instructions = make_file_instructions(
            name=self.dataset_name,
            split_infos=[self],
            instruction=str(self.name),
        )
        return instructions.file_instructions


@dataclass
class SubSplitInfo:
    """Wrapper around a sub split info.
    This class expose info on the subsplit:
    ```
    ds, info = datasets.load_dataset(..., split='train[75%:]', with_info=True)
    info.splits['train[75%:]'].num_examples
    ```
    """

    instructions: FileInstructions

    @property
    def num_examples(self):
        """Returns the number of example in the subsplit."""
        return self.instructions.num_examples

    @property
    def file_instructions(self):
        """Returns the list of dict(filename, take, skip)."""
        return self.instructions.file_instructions


class SplitBase(metaclass=abc.ABCMeta):
    # pylint: disable=line-too-long
    """Abstract base class for Split compositionality.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.

    There are three parts to the composition:
        1) The splits are composed (defined, merged, split,...) together before
             calling the `.as_dataset()` function. This is done with the `__add__`,
             `__getitem__`, which return a tree of `SplitBase` (whose leaf
             are the `NamedSplit` objects)

        ```
        split = datasets.Split.TRAIN + datasets.Split.TEST.subsplit(datasets.percent[:50])
        ```

        2) The `SplitBase` is forwarded to the `.as_dataset()` function
             to be resolved into actual read instruction. This is done by the
             `.get_read_instruction()` method which takes the real dataset splits
             (name, number of shards,...) and parse the tree to return a
             `SplitReadInstruction()` object

        ```
        read_instruction = split.get_read_instruction(self.info.splits)
        ```

        3) The `SplitReadInstruction` is then used in the `tf.data.Dataset` pipeline
             to define which files to read and how to skip examples within file.

    """

    # pylint: enable=line-too-long

    @abc.abstractmethod
    def get_read_instruction(self, split_dict):
        """Parse the descriptor tree and compile all read instructions together.

        Args:
            split_dict: `dict`, The `dict[split_name, SplitInfo]` of the dataset

        Returns:
            split_read_instruction: `SplitReadInstruction`
        """
        raise NotImplementedError("Abstract method")

    def __eq__(self, other):
        """Equality: datasets.Split.TRAIN == 'train'."""
        if isinstance(other, (NamedSplit, str)):
            return False
        raise NotImplementedError("Equality is not implemented between merged/sub splits.")

    def __ne__(self, other):
        """InEquality: datasets.Split.TRAIN != 'test'."""
        return not self.__eq__(other)

    def __add__(self, other):
        """Merging: datasets.Split.TRAIN + datasets.Split.TEST."""
        return _SplitMerged(self, other)

    def subsplit(self, arg=None, k=None, percent=None, weighted=None):  # pylint: disable=redefined-outer-name
        """Divides this split into subsplits.

        There are 3 ways to define subsplits, which correspond to the 3
        arguments `k` (get `k` even subsplits), `percent` (get a slice of the
        dataset with `datasets.percent`), and `weighted` (get subsplits with proportions
        specified by `weighted`).

        Example::

        ```
        # 50% train, 50% test
        train, test = split.subsplit(k=2)
        # 50% train, 25% test, 25% validation
        train, test, validation = split.subsplit(weighted=[2, 1, 1])
        # Extract last 20%
        subsplit = split.subsplit(datasets.percent[-20:])
        ```

        Warning: k and weighted will be converted into percent which mean that
        values below the percent will be rounded up or down. The final split may be
        bigger to deal with remainders. For instance:

        ```
        train, test, valid = split.subsplit(k=3)  # 33%, 33%, 34%
        s1, s2, s3, s4 = split.subsplit(weighted=[2, 2, 1, 1])  # 33%, 33%, 16%, 18%
        ```

        Args:
            arg: If no kwargs are given, `arg` will be interpreted as one of
                `k`, `percent`, or `weighted` depending on the type.
                For example:
                ```
                split.subsplit(10)  # Equivalent to split.subsplit(k=10)
                split.subsplit(datasets.percent[:-20])  # percent=datasets.percent[:-20]
                split.subsplit([1, 1, 2])  # weighted=[1, 1, 2]
                ```
            k: `int` If set, subdivide the split into `k` equal parts.
            percent: `datasets.percent slice`, return a single subsplit corresponding to
                a slice of the original split. For example:
                `split.subsplit(datasets.percent[-20:])  # Last 20% of the dataset`.
            weighted: `list[int]`, return a list of subsplits whose proportions match
                the normalized sum of the list. For example:
                `split.subsplit(weighted=[1, 1, 2])  # 25%, 25%, 50%`.

        Returns:
            A subsplit or list of subsplits extracted from this split object.
        """
        # Note that the percent kwargs redefine the outer name datasets.percent. This
        # is done for consistency (.subsplit(percent=datasets.percent[:40]))
        if sum(bool(x) for x in (arg, k, percent, weighted)) != 1:
            raise ValueError("Only one argument of subsplit should be set.")

        # Auto deduce k
        if isinstance(arg, int):
            k = arg
        elif isinstance(arg, slice):
            percent = arg
        elif isinstance(arg, list):
            weighted = arg

        if not (k or percent or weighted):
            raise ValueError(
                f"Invalid split argument {arg}. Only list, slice and int supported. "
                "One of k, weighted or percent should be set to a non empty value."
            )

        def assert_slices_coverage(slices):
            # Ensure that the expended slices cover all percents.
            assert sum((list(range(*s.indices(100))) for s in slices), []) == list(range(100))

        if k:
            if not 0 < k <= 100:
                raise ValueError(f"Subsplit k should be between 0 and 100, got {k}")
            shift = 100 // k
            slices = [slice(i * shift, (i + 1) * shift) for i in range(k)]
            # Round up last element to ensure all elements are taken
            slices[-1] = slice(slices[-1].start, 100)
            # Internal check to ensure full coverage
            assert_slices_coverage(slices)
            return tuple(_SubSplit(self, s) for s in slices)
        elif percent:
            return _SubSplit(self, percent)
        elif weighted:
            # Normalize the weighted sum
            total = sum(weighted)
            weighted = [100 * x // total for x in weighted]
            # Create the slice for each of the elements
            start = 0
            stop = 0
            slices = []
            for v in weighted:
                stop += v
                slices.append(slice(start, stop))
                start = stop
            # Round up last element to ensure all elements are taken
            slices[-1] = slice(slices[-1].start, 100)
            # Internal check to ensure full coverage
            assert_slices_coverage(slices)
            return tuple(_SubSplit(self, s) for s in slices)
        else:
            # Should not be possible
            raise ValueError("Could not determine the split")


# 2 requirements:
# 1. datasets.percent be sliceable
# 2. datasets.percent be documented
#
# Instances are not documented, so we want datasets.percent to be a class, but to
# have it be sliceable, we need this metaclass.
class PercentSliceMeta(type):
    def __getitem__(cls, slice_value):
        if not isinstance(slice_value, slice):
            raise ValueError(f"datasets.percent should only be called with slice, not {slice_value}")
        return slice_value


class PercentSlice(metaclass=PercentSliceMeta):
    # pylint: disable=line-too-long
    """Syntactic sugar for defining slice subsplits: `datasets.percent[75:-5]`.

    See the
    [guide on splits](../loading#slice-splits)
    for more information.
    """

    # pylint: enable=line-too-long
    pass


percent = PercentSlice  # pylint: disable=invalid-name


class _SplitMerged(SplitBase):
    """Represent two split descriptors merged together."""

    def __init__(self, split1, split2):
        self._split1 = split1
        self._split2 = split2

    def get_read_instruction(self, split_dict):
        read_instruction1 = self._split1.get_read_instruction(split_dict)
        read_instruction2 = self._split2.get_read_instruction(split_dict)
        return read_instruction1 + read_instruction2

    def __repr__(self):
        return f"({repr(self._split1)} + {repr(self._split2)})"


class _SubSplit(SplitBase):
    """Represent a sub split of a split descriptor."""

    def __init__(self, split, slice_value):
        self._split = split
        self._slice_value = slice_value

    def get_read_instruction(self, split_dict):
        return self._split.get_read_instruction(split_dict)[self._slice_value]

    def __repr__(self):
        slice_str = "{start}:{stop}"
        if self._slice_value.step is not None:
            slice_str += ":{step}"
        slice_str = slice_str.format(
            start="" if self._slice_value.start is None else self._slice_value.start,
            stop="" if self._slice_value.stop is None else self._slice_value.stop,
            step=self._slice_value.step,
        )
        return f"{repr(self._split)}(datasets.percent[{slice_str}])"


class NamedSplit(SplitBase):
    """Descriptor corresponding to a named split (train, test, ...).

    Example:
        Each descriptor can be composed with other using addition or slice:

            ```py
            split = datasets.Split.TRAIN.subsplit(datasets.percent[0:25]) + datasets.Split.TEST
            ```

        The resulting split will correspond to 25% of the train split merged with
        100% of the test split.

        A split cannot be added twice, so the following will fail:

            ```py
            split = (
                    datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
                    datasets.Split.TRAIN.subsplit(datasets.percent[75:])
            )  # Error
            split = datasets.Split.TEST + datasets.Split.ALL  # Error
            ```

        The slices can be applied only one time. So the following are valid:

            ```py
            split = (
                    datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
                    datasets.Split.TEST.subsplit(datasets.percent[:50])
            )
            split = (datasets.Split.TRAIN + datasets.Split.TEST).subsplit(datasets.percent[:50])
            ```

        But this is not valid:

            ```py
            train = datasets.Split.TRAIN
            test = datasets.Split.TEST
            split = train.subsplit(datasets.percent[:25]).subsplit(datasets.percent[:25])
            split = (train.subsplit(datasets.percent[:25]) + test).subsplit(datasets.percent[:50])
            ```
    """

    def __init__(self, name):
        self._name = name
        split_names_from_instruction = [split_instruction.split("[")[0] for split_instruction in name.split("+")]
        for split_name in split_names_from_instruction:
            if not re.match(_split_re, split_name):
                raise ValueError(f"Split name should match '{_split_re}' but got '{split_name}'.")

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"NamedSplit({self._name!r})"

    def __eq__(self, other):
        """Equality: datasets.Split.TRAIN == 'train'."""
        if isinstance(other, NamedSplit):
            return self._name == other._name  # pylint: disable=protected-access
        elif isinstance(other, SplitBase):
            return False
        elif isinstance(other, str):  # Other should be string
            return self._name == other
        else:
            return False

    def __lt__(self, other):
        return self._name < other._name  # pylint: disable=protected-access

    def __hash__(self):
        return hash(self._name)

    def get_read_instruction(self, split_dict):
        return SplitReadInstruction(split_dict[self._name])


class NamedSplitAll(NamedSplit):
    """Split corresponding to the union of all defined dataset splits."""

    def __init__(self):
        super().__init__("all")

    def __repr__(self):
        return "NamedSplitAll()"

    def get_read_instruction(self, split_dict):
        # Merge all dataset split together
        read_instructions = [SplitReadInstruction(s) for s in split_dict.values()]
        return sum(read_instructions, SplitReadInstruction())


class Split:
    # pylint: disable=line-too-long
    """`Enum` for dataset splits.

    Datasets are typically split into different subsets to be used at various
    stages of training and evaluation.

    - `TRAIN`: the training data.
    - `VALIDATION`: the validation data. If present, this is typically used as
      evaluation data while iterating on a model (e.g. changing hyperparameters,
      model architecture, etc.).
    - `TEST`: the testing data. This is the data to report metrics on. Typically
      you do not want to use this during model iteration as you may overfit to it.
    - `ALL`: the union of all defined dataset splits.

    All splits, including compositions inherit from `datasets.SplitBase`.

    See the [guide](../load_hub#splits) on splits for more information.

    Example:

    ```py
    >>> datasets.SplitGenerator(
    ...     name=datasets.Split.TRAIN,
    ...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and extract(url)},
    ... ),
    ... datasets.SplitGenerator(
    ...     name=datasets.Split.VALIDATION,
    ...     gen_kwargs={"split_key": "validation", "files": dl_manager.download_and extract(url)},
    ... ),
    ... datasets.SplitGenerator(
    ...     name=datasets.Split.TEST,
    ...     gen_kwargs={"split_key": "test", "files": dl_manager.download_and extract(url)},
    ... )
    ```
    """

    # pylint: enable=line-too-long
    TRAIN = NamedSplit("train")
    TEST = NamedSplit("test")
    VALIDATION = NamedSplit("validation")
    ALL = NamedSplitAll()

    def __new__(cls, name):
        """Create a custom split with datasets.Split('custom_name')."""
        return NamedSplitAll() if name == "all" else NamedSplit(name)


# Similar to SplitInfo, but contain an additional slice info
SlicedSplitInfo = collections.namedtuple(
    "SlicedSplitInfo",
    [
        "split_info",
        "slice_value",
    ],
)  # noqa: E231


class SplitReadInstruction:
    """Object containing the reading instruction for the dataset.

    Similarly to `SplitDescriptor` nodes, this object can be composed with itself,
    but the resolution happens instantaneously, instead of keeping track of the
    tree, such as all instructions are compiled and flattened in a single
    SplitReadInstruction object containing the list of files and slice to use.

    Once resolved, the instructions can be accessed with:

    ```
    read_instructions.get_list_sliced_split_info()  # List of splits to use
    ```

    """

    def __init__(self, split_info=None):
        self._splits = NonMutableDict(error_msg="Overlap between splits. Split {key} has been added with " "itself.")

        if split_info:
            self.add(SlicedSplitInfo(split_info=split_info, slice_value=None))

    def add(self, sliced_split):
        """Add a SlicedSplitInfo the read instructions."""
        # TODO(epot): Check that the number of examples per shard % 100 == 0
        # Otherwise the slices value may be unbalanced and not exactly reflect the
        # requested slice.
        self._splits[sliced_split.split_info.name] = sliced_split

    def __add__(self, other):
        """Merging split together."""
        # Will raise error if a split has already be added (NonMutableDict)
        # TODO(epot): If a split is already added but there is no overlap between
        # the slices, should merge the slices (ex: [:10] + [80:])
        split_instruction = SplitReadInstruction()
        split_instruction._splits.update(self._splits)  # pylint: disable=protected-access
        split_instruction._splits.update(other._splits)  # pylint: disable=protected-access
        return split_instruction

    def __getitem__(self, slice_value):
        """Sub-splits."""
        # Will raise an error if a split has already been sliced
        split_instruction = SplitReadInstruction()
        for v in self._splits.values():
            if v.slice_value is not None:
                raise ValueError(f"Trying to slice Split {v.split_info.name} which has already been sliced")
            v = v._asdict()
            v["slice_value"] = slice_value
            split_instruction.add(SlicedSplitInfo(**v))
        return split_instruction

    def get_list_sliced_split_info(self):
        return list(self._splits.values())


class SplitDict(dict):
    """Split info object."""

    def __init__(self, *args, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name

    def __getitem__(self, key: Union[SplitBase, str]):
        # 1st case: The key exists: `info.splits['train']`
        if str(key) in self:
            return super().__getitem__(str(key))
        # 2nd case: Uses instructions: `info.splits['train[50%]']`
        else:
            instructions = make_file_instructions(
                name=self.dataset_name,
                split_infos=self.values(),
                instruction=key,
            )
            return SubSplitInfo(instructions)

    def __setitem__(self, key: Union[SplitBase, str], value: SplitInfo):
        if key != value.name:
            raise ValueError(f"Cannot add elem. (key mismatch: '{key}' != '{value.name}')")
        super().__setitem__(key, value)

    def add(self, split_info: SplitInfo):
        """Add the split info."""
        if split_info.name in self:
            raise ValueError(f"Split {split_info.name} already present")
        split_info.dataset_name = self.dataset_name
        super().__setitem__(split_info.name, split_info)

    @property
    def total_num_examples(self):
        """Return the total number of examples."""
        return sum(s.num_examples for s in self.values())

    @classmethod
    def from_split_dict(cls, split_infos: Union[List, Dict], dataset_name: Optional[str] = None):
        """Returns a new SplitDict initialized from a Dict or List of `split_infos`."""
        if isinstance(split_infos, dict):
            split_infos = list(split_infos.values())

        if dataset_name is None:
            dataset_name = split_infos[0].get("dataset_name") if split_infos else None

        split_dict = cls(dataset_name=dataset_name)

        for split_info in split_infos:
            if isinstance(split_info, dict):
                split_info = SplitInfo(**split_info)
            split_dict.add(split_info)

        return split_dict

    def to_split_dict(self):
        """Returns a list of SplitInfo protos that we have."""
        out = []
        for split_name, split_info in self.items():
            split_info = copy.deepcopy(split_info)
            split_info.name = split_name
            out.append(split_info)
        return out

    def copy(self):
        return SplitDict.from_split_dict(self.to_split_dict(), self.dataset_name)

    def _to_yaml_list(self) -> list:
        out = [asdict(s) for s in self.to_split_dict()]
        # we don't need the shard lengths in YAML, since it depends on max_shard_size and num_proc
        for split_info_dict in out:
            split_info_dict.pop("shard_lengths", None)
        # we don't need the dataset_name attribute that is deprecated
        for split_info_dict in out:
            split_info_dict.pop("dataset_name", None)
        return out

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "SplitDict":
        return cls.from_split_dict(yaml_data)


@dataclass
class SplitGenerator:
    """Defines the split information for the generator.

    This should be used as returned value of
    `GeneratorBasedBuilder._split_generators`.
    See `GeneratorBasedBuilder._split_generators` for more info and example
    of usage.

    Args:
        name (`str`):
            Name of the `Split` for which the generator will
            create the examples.
        **gen_kwargs (additional keyword arguments):
            Keyword arguments to forward to the `DatasetBuilder._generate_examples` method
            of the builder.

    Example:

    ```py
    >>> datasets.SplitGenerator(
    ...     name=datasets.Split.TRAIN,
    ...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and_extract(url)},
    ... )
    ```
    """

    name: str
    gen_kwargs: Dict = dataclasses.field(default_factory=dict)
    split_info: SplitInfo = dataclasses.field(init=False)

    def __post_init__(self):
        self.name = str(self.name)  # Make sure we convert NamedSplits in strings
        NamedSplit(self.name)  # check that it's a valid split name
        self.split_info = SplitInfo(name=self.name)
