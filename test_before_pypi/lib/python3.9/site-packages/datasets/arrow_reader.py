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
"""Arrow ArrowReader."""

import copy
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map

from .download.download_config import DownloadConfig
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import cached_path


if TYPE_CHECKING:
    from .info import DatasetInfo  # noqa: F401
    from .splits import Split, SplitInfo  # noqa: F401


logger = logging.get_logger(__name__)

HF_GCP_BASE_URL = "https://storage.googleapis.com/huggingface-nlp/cache/datasets"

_SUB_SPEC_RE = re.compile(
    rf"""
^
 (?P<split>{_split_re[1:-1]})
 (\[
    ((?P<from>-?\d+)
     (?P<from_pct>%)?)?
    :
    ((?P<to>-?\d+)
     (?P<to_pct>%)?)?
 \])?(\((?P<rounding>[^\)]*)\))?
$
""",  # remove ^ and $
    re.X,
)

_ADDITION_SEP_RE = re.compile(r"\s*\+\s*")


class DatasetNotOnHfGcsError(ConnectionError):
    """When you can't get the dataset from the Hf google cloud storage"""

    pass


class MissingFilesOnHfGcsError(ConnectionError):
    """When some files are missing on the Hf oogle cloud storage"""

    pass


@dataclass(frozen=True)
class FileInstructions:
    """The file instructions associated with a split ReadInstruction.

    Attributes:
        num_examples: `int`, The total number of examples
        file_instructions: List[dict(filename, skip, take)], the files information.
            The filenames contains the relative path, not absolute.
            skip/take indicates which example read in the file: `ds.slice(skip, take)`
    """

    num_examples: int
    file_instructions: List[dict]


def make_file_instructions(
    name: str,
    split_infos: List["SplitInfo"],
    instruction: Union[str, "ReadInstruction"],
    filetype_suffix: Optional[str] = None,
    prefix_path: Optional[str] = None,
) -> FileInstructions:
    """Returns instructions of the split dict.

    Args:
        name (`str`): Name of the dataset.
        split_infos (`list` of `[SplitInfo]`): Dataset splits information.
        instruction ([`ReadInstruction`] or `str`): Reading instruction for a dataset.
        filetype_suffix (`str`, *optional*): Suffix of dataset files, e.g. 'arrow' or 'parquet'.
        prefix_path (`str`, *optional*): Prefix of dataset files, e.g. directory name.

    Returns:
        [`FileInstructions`]
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected str 'name', but got: {type(name).__name__}")
    elif not name:
        raise ValueError("Expected non-empty str 'name'")
    name2len = {info.name: info.num_examples for info in split_infos}
    name2shard_lengths = {info.name: info.shard_lengths for info in split_infos}
    name2filenames = {
        info.name: filenames_for_dataset_split(
            path=prefix_path,
            dataset_name=name,
            split=info.name,
            filetype_suffix=filetype_suffix,
            shard_lengths=name2shard_lengths[info.name],
        )
        for info in split_infos
    }
    if not isinstance(instruction, ReadInstruction):
        instruction = ReadInstruction.from_spec(instruction)
    # Create the absolute instruction (per split)
    absolute_instructions = instruction.to_absolute(name2len)

    # For each split, return the files instruction (skip/take)
    file_instructions = []
    num_examples = 0
    for abs_instr in absolute_instructions:
        split_length = name2len[abs_instr.splitname]
        filenames = name2filenames[abs_instr.splitname]
        shard_lengths = name2shard_lengths[abs_instr.splitname]
        from_ = 0 if abs_instr.from_ is None else abs_instr.from_
        to = split_length if abs_instr.to is None else abs_instr.to
        if shard_lengths is None:  # not sharded
            for filename in filenames:
                take = to - from_
                if take == 0:
                    continue
                num_examples += take
                file_instructions.append({"filename": filename, "skip": from_, "take": take})
        else:  # sharded
            index_start = 0  # Beginning (included) of moving window.
            index_end = 0  # End (excluded) of moving window.
            for filename, shard_length in zip(filenames, shard_lengths):
                index_end += shard_length
                if from_ < index_end and to > index_start:  # There is something to take.
                    skip = from_ - index_start if from_ > index_start else 0
                    take = to - index_start - skip if to < index_end else -1
                    if take == 0:
                        continue
                    file_instructions.append({"filename": filename, "skip": skip, "take": take})
                    num_examples += shard_length - skip if take == -1 else take
                index_start += shard_length
    return FileInstructions(
        num_examples=num_examples,
        file_instructions=file_instructions,
    )


class BaseReader:
    """
    Build a Dataset object out of Instruction instance(s).
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes ArrowReader.

        Args:
            path (str): path where tfrecords are stored.
            info (DatasetInfo): info about the dataset.
        """
        self._path: str = path
        self._info: Optional["DatasetInfo"] = info
        self._filetype_suffix: Optional[str] = None

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        """Returns a Dataset instance from given (filename, skip, take)."""
        raise NotImplementedError

    def _read_files(self, files, in_memory=False) -> Table:
        """Returns Dataset for given file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contain the absolute path, not relative.
                skip/take indicates which example read in the file: `ds.slice(skip, take)`
            in_memory (bool, default False): Whether to copy the data in-memory.
        """
        if len(files) == 0 or not all(isinstance(f, dict) for f in files):
            raise ValueError("please provide valid file informations")
        files = copy.deepcopy(files)
        for f in files:
            f["filename"] = os.path.join(self._path, f["filename"])

        pa_tables = thread_map(
            partial(self._get_table_from_filename, in_memory=in_memory),
            files,
            tqdm_class=hf_tqdm,
            desc="Loading dataset shards",
            # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
            disable=len(files) <= 16 or None,
        )
        pa_tables = [t for t in pa_tables if len(t) > 0]
        if not pa_tables and (self._info is None or self._info.features is None):
            raise ValueError(
                "Tried to read an empty table. Please specify at least info.features to create an empty table with the right type."
            )
        pa_tables = pa_tables or [InMemoryTable.from_batches([], schema=pa.schema(self._info.features.type))]
        pa_table = concat_tables(pa_tables) if len(pa_tables) != 1 else pa_tables[0]
        return pa_table

    def get_file_instructions(self, name, instruction, split_infos):
        """Return list of dict {'filename': str, 'skip': int, 'take': int}"""
        file_instructions = make_file_instructions(
            name, split_infos, instruction, filetype_suffix=self._filetype_suffix, prefix_path=self._path
        )
        files = file_instructions.file_instructions
        return files

    def read(
        self,
        name,
        instructions,
        split_infos,
        in_memory=False,
    ):
        """Returns Dataset instance(s).

        Args:
            name (str): name of the dataset.
            instructions (ReadInstruction): instructions to read.
                Instruction can be string and will then be passed to the Instruction
                constructor as it.
            split_infos (list of SplitInfo proto): the available splits for dataset.
            in_memory (bool, default False): Whether to copy the data in-memory.

        Returns:
             kwargs to build a single Dataset instance.
        """

        files = self.get_file_instructions(name, instructions, split_infos)
        if not files:
            msg = f'Instruction "{instructions}" corresponds to no data!'
            raise ValueError(msg)
        return self.read_files(files=files, original_instructions=instructions, in_memory=in_memory)

    def read_files(
        self,
        files: List[dict],
        original_instructions: Union[None, "ReadInstruction", "Split"] = None,
        in_memory=False,
    ):
        """Returns single Dataset instance for the set of file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contains the relative path, not absolute.
                skip/take indicates which example read in the file: `ds.skip().take()`
            original_instructions: store the original instructions used to build the dataset split in the dataset.
            in_memory (bool, default False): Whether to copy the data in-memory.

        Returns:
            kwargs to build a Dataset instance.
        """
        # Prepend path to filename
        pa_table = self._read_files(files, in_memory=in_memory)
        # If original_instructions is not None, convert it to a human-readable NamedSplit
        if original_instructions is not None:
            from .splits import Split  # noqa

            split = Split(str(original_instructions))
        else:
            split = None
        dataset_kwargs = {"arrow_table": pa_table, "info": self._info, "split": split}
        return dataset_kwargs

    @deprecated()
    def download_from_hf_gcs(self, download_config: DownloadConfig, relative_data_dir):
        """
        Download the dataset files from the Hf GCS

        Args:
            dl_cache_dir: `str`, the local cache directory used to download files
            relative_data_dir: `str`, the relative directory of the remote files from
                the `datasets` directory on GCS.

        """
        remote_cache_dir = HF_GCP_BASE_URL + "/" + relative_data_dir.replace(os.sep, "/")
        try:
            remote_dataset_info = os.path.join(remote_cache_dir, "dataset_info.json")
            downloaded_dataset_info = cached_path(
                remote_dataset_info.replace(os.sep, "/"), download_config=download_config
            )
            shutil.move(downloaded_dataset_info, os.path.join(self._path, "dataset_info.json"))
            if self._info is not None:
                self._info.update(self._info.from_directory(self._path))
        except FileNotFoundError as err:
            raise DatasetNotOnHfGcsError(err) from None
        try:
            for split in self._info.splits:
                file_instructions = self.get_file_instructions(
                    name=self._info.builder_name,
                    instruction=split,
                    split_infos=self._info.splits.values(),
                )
                for file_instruction in file_instructions:
                    file_to_download = str(Path(file_instruction["filename"]).relative_to(self._path))
                    remote_prepared_filename = os.path.join(remote_cache_dir, file_to_download)
                    downloaded_prepared_filename = cached_path(
                        remote_prepared_filename.replace(os.sep, "/"), download_config=download_config
                    )
                    shutil.move(downloaded_prepared_filename, file_instruction["filename"])
        except FileNotFoundError as err:
            raise MissingFilesOnHfGcsError(err) from None


class ArrowReader(BaseReader):
    """
    Build a Dataset object out of Instruction instance(s).
    This Reader uses either memory mapping or file descriptors (in-memory) on arrow files.
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes ArrowReader.

        Args:
            path (str): path where Arrow files are stored.
            info (DatasetInfo): info about the dataset.
        """
        super().__init__(path, info)
        self._filetype_suffix = "arrow"

    def _get_table_from_filename(self, filename_skip_take, in_memory=False) -> Table:
        """Returns a Dataset instance from given (filename, skip, take)."""
        filename, skip, take = (
            filename_skip_take["filename"],
            filename_skip_take["skip"] if "skip" in filename_skip_take else None,
            filename_skip_take["take"] if "take" in filename_skip_take else None,
        )
        table = ArrowReader.read_table(filename, in_memory=in_memory)
        if take == -1:
            take = len(table) - skip
        # here we don't want to slice an empty table, or it may segfault
        if skip is not None and take is not None and not (skip == 0 and take == len(table)):
            table = table.slice(skip, take)
        return table

    @staticmethod
    def read_table(filename, in_memory=False) -> Table:
        """
        Read table from file.

        Args:
            filename (str): File name of the table.
            in_memory (bool, default=False): Whether to copy the data in-memory.

        Returns:
            pyarrow.Table
        """
        table_cls = InMemoryTable if in_memory else MemoryMappedTable
        return table_cls.from_file(filename)


class ParquetReader(BaseReader):
    """
    Build a Dataset object out of Instruction instance(s).
    This Reader uses memory mapping on parquet files.
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes ParquetReader.

        Args:
            path (str): path where tfrecords are stored.
            info (DatasetInfo): info about the dataset.
        """
        super().__init__(path, info)
        self._filetype_suffix = "parquet"

    def _get_table_from_filename(self, filename_skip_take, **kwargs):
        """Returns a Dataset instance from given (filename, skip, take)."""
        filename, skip, take = (
            filename_skip_take["filename"],
            filename_skip_take["skip"] if "skip" in filename_skip_take else None,
            filename_skip_take["take"] if "take" in filename_skip_take else None,
        )
        # Parquet read_table always loads data in memory, independently of memory_map
        pa_table = pq.read_table(filename, memory_map=True)
        # here we don't want to slice an empty table, or it may segfault
        if skip is not None and take is not None and not (skip == 0 and take == len(pa_table)):
            pa_table = pa_table.slice(skip, take)
        return pa_table


@dataclass(frozen=True)
class _AbsoluteInstruction:
    """A machine friendly slice: defined absolute positive boundaries."""

    splitname: str
    from_: int  # uint (starting index).
    to: int  # uint (ending index).


@dataclass(frozen=True)
class _RelativeInstruction:
    """Represents a single parsed slicing instruction, can use % and negatives."""

    splitname: str
    from_: Optional[int] = None  # int (starting index) or None if no lower boundary.
    to: Optional[int] = None  # int (ending index) or None if no upper boundary.
    unit: Optional[str] = None
    rounding: Optional[str] = None

    def __post_init__(self):
        if self.unit is not None and self.unit not in ["%", "abs"]:
            raise ValueError("unit must be either % or abs")
        if self.rounding is not None and self.rounding not in ["closest", "pct1_dropremainder"]:
            raise ValueError("rounding must be either closest or pct1_dropremainder")
        if self.unit != "%" and self.rounding is not None:
            raise ValueError("It is forbidden to specify rounding if not using percent slicing.")
        if self.unit == "%" and self.from_ is not None and abs(self.from_) > 100:
            raise ValueError("Percent slice boundaries must be > -100 and < 100.")
        if self.unit == "%" and self.to is not None and abs(self.to) > 100:
            raise ValueError("Percent slice boundaries must be > -100 and < 100.")
        # Update via __dict__ due to instance being "frozen"
        self.__dict__["rounding"] = "closest" if self.rounding is None and self.unit == "%" else self.rounding


def _str_to_read_instruction(spec):
    """Returns ReadInstruction for given string."""
    res = _SUB_SPEC_RE.match(spec)
    if not res:
        raise ValueError(f"Unrecognized instruction format: {spec}")
    unit = "%" if res.group("from_pct") or res.group("to_pct") else "abs"
    return ReadInstruction(
        split_name=res.group("split"),
        rounding=res.group("rounding"),
        from_=int(res.group("from")) if res.group("from") else None,
        to=int(res.group("to")) if res.group("to") else None,
        unit=unit,
    )


def _pct_to_abs_pct1(boundary, num_examples):
    # Using math.trunc here, since -99.5% should give -99%, not -100%.
    if num_examples < 100:
        msg = (
            'Using "pct1_dropremainder" rounding on a split with less than 100 '
            "elements is forbidden: it always results in an empty dataset."
        )
        raise ValueError(msg)
    return boundary * math.trunc(num_examples / 100.0)


def _pct_to_abs_closest(boundary, num_examples):
    return int(round(boundary * num_examples / 100.0))


def _rel_to_abs_instr(rel_instr, name2len):
    """Returns _AbsoluteInstruction instance for given RelativeInstruction.

    Args:
        rel_instr: RelativeInstruction instance.
        name2len: dict {split_name: num_examples}.
    """
    pct_to_abs = _pct_to_abs_closest if rel_instr.rounding == "closest" else _pct_to_abs_pct1
    split = rel_instr.splitname
    if split not in name2len:
        raise ValueError(f'Unknown split "{split}". Should be one of {list(name2len)}.')
    num_examples = name2len[split]
    from_ = rel_instr.from_
    to = rel_instr.to
    if rel_instr.unit == "%":
        from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
        to = num_examples if to is None else pct_to_abs(to, num_examples)
    else:
        from_ = 0 if from_ is None else from_
        to = num_examples if to is None else to
    if from_ < 0:
        from_ = max(num_examples + from_, 0)
    if to < 0:
        to = max(num_examples + to, 0)
    from_ = min(from_, num_examples)
    to = min(to, num_examples)
    return _AbsoluteInstruction(split, from_, to)


class ReadInstruction:
    """Reading instruction for a dataset.

    Examples::

      # The following lines are equivalent:
      ds = datasets.load_dataset('mnist', split='test[:33%]')
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction.from_spec('test[:33%]'))
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction('test', to=33, unit='%'))
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction(
          'test', from_=0, to=33, unit='%'))

      # The following lines are equivalent:
      ds = datasets.load_dataset('mnist', split='test[:33%]+train[1:-1]')
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction.from_spec(
          'test[:33%]+train[1:-1]'))
      ds = datasets.load_dataset('mnist', split=(
          datasets.ReadInstruction('test', to=33, unit='%') +
          datasets.ReadInstruction('train', from_=1, to=-1, unit='abs')))

      # The following lines are equivalent:
      ds = datasets.load_dataset('mnist', split='test[:33%](pct1_dropremainder)')
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction.from_spec(
          'test[:33%](pct1_dropremainder)'))
      ds = datasets.load_dataset('mnist', split=datasets.ReadInstruction(
          'test', from_=0, to=33, unit='%', rounding="pct1_dropremainder"))

      # 10-fold validation:
      tests = datasets.load_dataset(
          'mnist',
          [datasets.ReadInstruction('train', from_=k, to=k+10, unit='%')
          for k in range(0, 100, 10)])
      trains = datasets.load_dataset(
          'mnist',
          [datasets.ReadInstruction('train', to=k, unit='%') + datasets.ReadInstruction('train', from_=k+10, unit='%')
          for k in range(0, 100, 10)])

    """

    def _init(self, relative_instructions):
        # Private initializer.
        self._relative_instructions = relative_instructions

    @classmethod
    def _read_instruction_from_relative_instructions(cls, relative_instructions):
        """Returns ReadInstruction obj initialized with relative_instructions."""
        # Use __new__ to bypass __init__ used by public API and not conveniant here.
        result = cls.__new__(cls)
        result._init(relative_instructions)  # pylint: disable=protected-access
        return result

    def __init__(self, split_name, rounding=None, from_=None, to=None, unit=None):
        """Initialize ReadInstruction.

        Args:
            split_name (str): name of the split to read. Eg: 'train'.
            rounding (str, optional): The rounding behaviour to use when percent slicing is
                used. Ignored when slicing with absolute indices.
                Possible values:
                 - 'closest' (default): The specified percentages are rounded to the
                     closest value. Use this if you want specified percents to be as
                     much exact as possible.
                 - 'pct1_dropremainder': the specified percentages are treated as
                     multiple of 1%. Use this option if you want consistency. Eg:
                         len(5%) == 5 * len(1%).
                     Using this option, one might not be able to use the full set of
                     examples, if the number of those is not a multiple of 100.
            from_ (int):
            to (int): alternative way of specifying slicing boundaries. If any of
                {from_, to, unit} argument is used, slicing cannot be specified as
                string.
            unit (str): optional, one of:
                '%': to set the slicing unit as percents of the split size.
                'abs': to set the slicing unit as absolute numbers.
        """
        # This constructor is not always called. See factory method
        # `_read_instruction_from_relative_instructions`. Common init instructions
        # MUST be placed in the _init method.
        self._init([_RelativeInstruction(split_name, from_, to, unit, rounding)])

    @classmethod
    def from_spec(cls, spec):
        """Creates a `ReadInstruction` instance out of a string spec.

        Args:
            spec (`str`):
                Split(s) + optional slice(s) to read + optional rounding
                if percents are used as the slicing unit. A slice can be specified,
                using absolute numbers (`int`) or percentages (`int`).

        Examples:

            ```
            test: test split.
            test + validation: test split + validation split.
            test[10:]: test split, minus its first 10 records.
            test[:10%]: first 10% records of test split.
            test[:20%](pct1_dropremainder): first 10% records, rounded with the pct1_dropremainder rounding.
            test[:-5%]+train[40%:60%]: first 95% of test + middle 20% of train.
            ```

        Returns:
            ReadInstruction instance.
        """
        spec = str(spec)  # Need to convert to str in case of NamedSplit instance.
        subs = _ADDITION_SEP_RE.split(spec)
        if not subs:
            raise ValueError(f"No instructions could be built out of {spec}")
        instruction = _str_to_read_instruction(subs[0])
        return sum((_str_to_read_instruction(sub) for sub in subs[1:]), instruction)

    def to_spec(self):
        rel_instr_specs = []
        for rel_instr in self._relative_instructions:
            rel_instr_spec = rel_instr.splitname
            if rel_instr.from_ is not None or rel_instr.to is not None:
                from_ = rel_instr.from_
                to = rel_instr.to
                unit = rel_instr.unit
                rounding = rel_instr.rounding
                unit = unit if unit == "%" else ""
                from_ = str(from_) + unit if from_ is not None else ""
                to = str(to) + unit if to is not None else ""
                slice_str = f"[{from_}:{to}]"
                rounding_str = (
                    f"({rounding})" if unit == "%" and rounding is not None and rounding != "closest" else ""
                )
                rel_instr_spec += slice_str + rounding_str
            rel_instr_specs.append(rel_instr_spec)
        return "+".join(rel_instr_specs)

    def __add__(self, other):
        """Returns a new ReadInstruction obj, result of appending other to self."""
        if not isinstance(other, ReadInstruction):
            msg = "ReadInstruction can only be added to another ReadInstruction obj."
            raise TypeError(msg)
        self_ris = self._relative_instructions
        other_ris = other._relative_instructions  # pylint: disable=protected-access
        if (
            self_ris[0].unit != "abs"
            and other_ris[0].unit != "abs"
            and self._relative_instructions[0].rounding != other_ris[0].rounding
        ):
            raise ValueError("It is forbidden to sum ReadInstruction instances with different rounding values.")
        return self._read_instruction_from_relative_instructions(self_ris + other_ris)

    def __str__(self):
        return self.to_spec()

    def __repr__(self):
        return f"ReadInstruction({self._relative_instructions})"

    def to_absolute(self, name2len):
        """Translate instruction into a list of absolute instructions.

        Those absolute instructions are then to be added together.

        Args:
            name2len (`dict`):
                Associating split names to number of examples.

        Returns:
            list of _AbsoluteInstruction instances (corresponds to the + in spec).
        """
        return [_rel_to_abs_instr(rel_instr, name2len) for rel_instr in self._relative_instructions]
