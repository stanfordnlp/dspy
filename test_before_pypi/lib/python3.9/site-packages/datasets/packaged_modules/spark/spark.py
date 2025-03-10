import os
import posixpath
import uuid
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

import datasets
from datasets.arrow_writer import ArrowWriter, ParquetWriter
from datasets.config import MAX_SHARD_SIZE
from datasets.filesystems import (
    is_remote_filesystem,
    rename,
)
from datasets.iterable_dataset import _BaseExamplesIterable
from datasets.utils import experimental
from datasets.utils.py_utils import convert_file_size_to_int


logger = datasets.utils.logging.get_logger(__name__)

if TYPE_CHECKING:
    import pyspark
    import pyspark.sql


@dataclass
class SparkConfig(datasets.BuilderConfig):
    """BuilderConfig for Spark."""

    features: Optional[datasets.Features] = None

    def __post_init__(self):
        super().__post_init__()


def _reorder_dataframe_by_partition(df: "pyspark.sql.DataFrame", new_partition_order: List[int]):
    df_combined = df.select("*").where(f"part_id = {new_partition_order[0]}")
    for partition_id in new_partition_order[1:]:
        partition_df = df.select("*").where(f"part_id = {partition_id}")
        df_combined = df_combined.union(partition_df)
    return df_combined


def _generate_iterable_examples(
    df: "pyspark.sql.DataFrame",
    partition_order: List[int],
    state_dict: Optional[dict] = None,
):
    import pyspark

    df_with_partition_id = df.select("*", pyspark.sql.functions.spark_partition_id().alias("part_id"))
    partition_idx_start = state_dict["partition_idx"] if state_dict else 0
    partition_df = _reorder_dataframe_by_partition(df_with_partition_id, partition_order[partition_idx_start:])
    # pipeline next partition in parallel to hide latency
    rows = partition_df.toLocalIterator(prefetchPartitions=True)
    curr_partition = None
    row_id = state_dict["partition_example_idx"] if state_dict else 0
    for row in islice(rows, row_id, None):
        row_as_dict = row.asDict()
        part_id = row_as_dict["part_id"]
        row_as_dict.pop("part_id")
        if curr_partition != part_id:
            if state_dict and curr_partition is not None:
                state_dict["partition_idx"] += 1
            curr_partition = part_id
            row_id = 0
        if state_dict:
            state_dict["partition_example_idx"] = row_id + 1
        yield f"{part_id}_{row_id}", row_as_dict
        row_id += 1


class SparkExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        df: "pyspark.sql.DataFrame",
        partition_order=None,
    ):
        super().__init__()
        self.df = df
        self.partition_order = partition_order or range(self.df.rdd.getNumPartitions())

    def _init_state_dict(self) -> dict:
        self._state_dict = {"partition_idx": 0, "partition_example_idx": 0}
        return self._state_dict

    @experimental
    def load_state_dict(self, state_dict: dict) -> dict:
        return super().load_state_dict(state_dict)

    def __iter__(self):
        yield from _generate_iterable_examples(self.df, self.partition_order, self._state_dict)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "SparkExamplesIterable":
        partition_order = list(range(self.df.rdd.getNumPartitions()))
        generator.shuffle(partition_order)
        return SparkExamplesIterable(self.df, partition_order=partition_order)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "SparkExamplesIterable":
        partition_order = self.split_shard_indices_by_worker(worker_id, num_workers)
        return SparkExamplesIterable(self.df, partition_order=partition_order)

    @property
    def n_shards(self) -> int:
        return len(self.partition_order)


class Spark(datasets.DatasetBuilder):
    BUILDER_CONFIG_CLASS = SparkConfig

    def __init__(
        self,
        df: "pyspark.sql.DataFrame",
        cache_dir: str = None,
        working_dir: str = None,
        **config_kwargs,
    ):
        import pyspark

        self._spark = pyspark.sql.SparkSession.builder.getOrCreate()
        self.df = df
        self._working_dir = working_dir

        super().__init__(
            cache_dir=cache_dir,
            config_name=str(self.df.semanticHash()),
            **config_kwargs,
        )

    def _validate_cache_dir(self):
        # Define this so that we don't reference self in create_cache_and_write_probe, which will result in a pickling
        # error due to pickling the SparkContext.
        cache_dir = self._cache_dir

        # Returns the path of the created file.
        def create_cache_and_write_probe(context):
            # makedirs with exist_ok will recursively create the directory. It will not throw an error if directories
            # already exist.
            os.makedirs(cache_dir, exist_ok=True)
            probe_file = os.path.join(cache_dir, "fs_test" + uuid.uuid4().hex)
            # Opening the file in append mode will create a new file unless it already exists, in which case it will not
            # change the file contents.
            open(probe_file, "a")
            return [probe_file]

        if self._spark.conf.get("spark.master", "").startswith("local"):
            return

        # If the cluster is multi-node, make sure that the user provided a cache_dir and that it is on an NFS
        # accessible to the driver.
        # TODO: Stream batches to the driver using ArrowCollectSerializer instead of throwing an error.
        if self._cache_dir:
            probe = (
                self._spark.sparkContext.parallelize(range(1), 1).mapPartitions(create_cache_and_write_probe).collect()
            )
            if os.path.isfile(probe[0]):
                return

        raise ValueError(
            "When using Dataset.from_spark on a multi-node cluster, the driver and all workers should be able to access cache_dir"
        )

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager: datasets.download.download_manager.DownloadManager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _repartition_df_if_needed(self, max_shard_size):
        import pyspark

        def get_arrow_batch_size(it):
            for batch in it:
                yield pa.RecordBatch.from_pydict({"batch_bytes": [batch.nbytes]})

        df_num_rows = self.df.count()
        sample_num_rows = df_num_rows if df_num_rows <= 100 else 100
        # Approximate the size of each row (in Arrow format) by averaging over a max-100-row sample.
        approx_bytes_per_row = (
            self.df.limit(sample_num_rows)
            .repartition(1)
            .mapInArrow(get_arrow_batch_size, "batch_bytes: long")
            .agg(pyspark.sql.functions.sum("batch_bytes").alias("sample_bytes"))
            .collect()[0]
            .sample_bytes
            / sample_num_rows
        )
        approx_total_size = approx_bytes_per_row * df_num_rows
        if approx_total_size > max_shard_size:
            # Make sure there is at least one row per partition.
            new_num_partitions = min(df_num_rows, int(approx_total_size / max_shard_size))
            self.df = self.df.repartition(new_num_partitions)

    def _prepare_split_single(
        self,
        fpath: str,
        file_format: str,
        max_shard_size: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        import pyspark

        writer_class = ParquetWriter if file_format == "parquet" else ArrowWriter
        working_fpath = os.path.join(self._working_dir, os.path.basename(fpath)) if self._working_dir else fpath
        embed_local_files = file_format == "parquet"

        # Define these so that we don't reference self in write_arrow, which will result in a pickling error due to
        # pickling the SparkContext.
        features = self.config.features
        writer_batch_size = self._writer_batch_size
        storage_options = self._fs.storage_options

        def write_arrow(it):
            # Within the same SparkContext, no two task attempts will share the same attempt ID.
            task_id = pyspark.TaskContext().taskAttemptId()
            first_batch = next(it, None)
            if first_batch is None:
                # Some partitions might not receive any data.
                return pa.RecordBatch.from_arrays(
                    [[task_id], [0], [0]],
                    names=["task_id", "num_examples", "num_bytes"],
                )
            shard_id = 0
            writer = writer_class(
                features=features,
                path=working_fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                writer_batch_size=writer_batch_size,
                storage_options=storage_options,
                embed_local_files=embed_local_files,
            )
            table = pa.Table.from_batches([first_batch])
            writer.write_table(table)
            for batch in it:
                if max_shard_size is not None and writer._num_bytes >= max_shard_size:
                    num_examples, num_bytes = writer.finalize()
                    writer.close()
                    yield pa.RecordBatch.from_arrays(
                        [[task_id], [num_examples], [num_bytes]],
                        names=["task_id", "num_examples", "num_bytes"],
                    )
                    shard_id += 1
                    writer = writer_class(
                        features=writer._features,
                        path=working_fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                        writer_batch_size=writer_batch_size,
                        storage_options=storage_options,
                        embed_local_files=embed_local_files,
                    )
                table = pa.Table.from_batches([batch])
                writer.write_table(table)

            if writer._num_bytes > 0:
                num_examples, num_bytes = writer.finalize()
                writer.close()
                yield pa.RecordBatch.from_arrays(
                    [[task_id], [num_examples], [num_bytes]],
                    names=["task_id", "num_examples", "num_bytes"],
                )

            if working_fpath != fpath:
                for file in os.listdir(os.path.dirname(working_fpath)):
                    dest = os.path.join(os.path.dirname(fpath), os.path.basename(file))
                    shutil.move(file, dest)

        stats = (
            self.df.mapInArrow(write_arrow, "task_id: long, num_examples: long, num_bytes: long")
            .groupBy("task_id")
            .agg(
                pyspark.sql.functions.sum("num_examples").alias("total_num_examples"),
                pyspark.sql.functions.sum("num_bytes").alias("total_num_bytes"),
                pyspark.sql.functions.count("num_bytes").alias("num_shards"),
                pyspark.sql.functions.collect_list("num_examples").alias("shard_lengths"),
            )
            .collect()
        )
        for row in stats:
            yield row.task_id, (row.total_num_examples, row.total_num_bytes, row.num_shards, row.shard_lengths)

    def _prepare_split(
        self,
        split_generator: "datasets.SplitGenerator",
        file_format: str = "arrow",
        max_shard_size: Optional[Union[str, int]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        self._validate_cache_dir()

        max_shard_size = convert_file_size_to_int(max_shard_size or MAX_SHARD_SIZE)
        self._repartition_df_if_needed(max_shard_size)
        is_local = not is_remote_filesystem(self._fs)
        path_join = os.path.join if is_local else posixpath.join

        SUFFIX = "-TTTTT-SSSSS-of-NNNNN"
        fname = f"{self.name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = path_join(self._output_dir, fname)

        total_num_examples = 0
        total_num_bytes = 0
        total_shards = 0
        task_id_and_num_shards = []
        all_shard_lengths = []

        for task_id, content in self._prepare_split_single(fpath, file_format, max_shard_size):
            (
                num_examples,
                num_bytes,
                num_shards,
                shard_lengths,
            ) = content
            if num_bytes > 0:
                total_num_examples += num_examples
                total_num_bytes += num_bytes
                total_shards += num_shards
                task_id_and_num_shards.append((task_id, num_shards))
                all_shard_lengths.extend(shard_lengths)

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            split_generator.split_info.shard_lengths = all_shard_lengths

            # Define fs outside of _rename_shard so that we don't reference self in the function, which will result in a
            # pickling error due to pickling the SparkContext.
            fs = self._fs

            # use the -SSSSS-of-NNNNN pattern
            def _rename_shard(
                task_id: int,
                shard_id: int,
                global_shard_id: int,
            ):
                rename(
                    fs,
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                    fpath.replace("TTTTT-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            args = []
            global_shard_id = 0
            for i in range(len(task_id_and_num_shards)):
                task_id, num_shards = task_id_and_num_shards[i]
                for shard_id in range(num_shards):
                    args.append([task_id, shard_id, global_shard_id])
                    global_shard_id += 1
            self._spark.sparkContext.parallelize(args, len(args)).map(lambda args: _rename_shard(*args)).collect()
        else:
            # don't use any pattern
            shard_id = 0
            task_id = task_id_and_num_shards[0][0]
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace("TTTTT", f"{task_id:05d}"),
                fpath.replace(SUFFIX, ""),
            )

    def _get_examples_iterable_for_split(
        self,
        split_generator: "datasets.SplitGenerator",
    ) -> SparkExamplesIterable:
        return SparkExamplesIterable(self.df)
