import itertools
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa

import datasets
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class ArrowConfig(datasets.BuilderConfig):
    """BuilderConfig for Arrow."""

    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()


class Arrow(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ArrowConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            # Use `dl_manager.iter_files` to skip hidden files in an extracted archive
            files = [dl_manager.iter_files(file) for file in files]
            # Infer features if they are stored in the arrow schema
            if self.info.features is None:
                for file in itertools.chain.from_iterable(files):
                    with open(file, "rb") as f:
                        try:
                            reader = pa.ipc.open_stream(f)
                        except pa.lib.ArrowInvalid:
                            reader = pa.ipc.open_file(f)
                    self.info.features = datasets.Features.from_arrow_schema(reader.schema)
                    break
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_tables(self, files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            with open(file, "rb") as f:
                try:
                    try:
                        batches = pa.ipc.open_stream(f)
                    except pa.lib.ArrowInvalid:
                        reader = pa.ipc.open_file(f)
                        batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
                    for batch_idx, record_batch in enumerate(batches):
                        pa_table = pa.Table.from_batches([record_batch])
                        # Uncomment for debugging (will print the Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                        yield f"{file_idx}_{batch_idx}", self._cast_table(pa_table)
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise
