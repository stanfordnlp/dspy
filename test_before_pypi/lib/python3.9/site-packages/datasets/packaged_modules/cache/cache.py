import glob
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pyarrow as pa

import datasets
import datasets.config
import datasets.data_files
from datasets.naming import camelcase_to_snakecase, filenames_for_dataset_split


logger = datasets.utils.logging.get_logger(__name__)


def _get_modification_time(cached_directory_path):
    return (Path(cached_directory_path)).stat().st_mtime


def _find_hash_in_cache(
    dataset_name: str,
    config_name: Optional[str],
    cache_dir: Optional[str],
    config_kwargs: dict,
    custom_features: Optional[datasets.Features],
) -> Tuple[str, str, str]:
    if config_name or config_kwargs or custom_features:
        config_id = datasets.BuilderConfig(config_name or "default").create_config_id(
            config_kwargs=config_kwargs, custom_features=custom_features
        )
    else:
        config_id = None
    cache_dir = os.path.expanduser(str(cache_dir or datasets.config.HF_DATASETS_CACHE))
    namespace_and_dataset_name = dataset_name.split("/")
    namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
    cached_relative_path = "___".join(namespace_and_dataset_name)
    cached_datasets_directory_path_root = os.path.join(cache_dir, cached_relative_path)
    cached_directory_paths = [
        cached_directory_path
        for cached_directory_path in glob.glob(
            os.path.join(cached_datasets_directory_path_root, config_id or "*", "*", "*")
        )
        if os.path.isdir(cached_directory_path)
        and (
            config_kwargs
            or custom_features
            or json.loads(Path(cached_directory_path, "dataset_info.json").read_text(encoding="utf-8"))["config_name"]
            == Path(cached_directory_path).parts[-3]  # no extra params => config_id == config_name
        )
    ]
    if not cached_directory_paths:
        cached_directory_paths = [
            cached_directory_path
            for cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", "*", "*"))
            if os.path.isdir(cached_directory_path)
        ]
        available_configs = sorted(
            {Path(cached_directory_path).parts[-3] for cached_directory_path in cached_directory_paths}
        )
        raise ValueError(
            f"Couldn't find cache for {dataset_name}"
            + (f" for config '{config_id}'" if config_id else "")
            + (f"\nAvailable configs in the cache: {available_configs}" if available_configs else "")
        )
    # get most recent
    cached_directory_path = Path(sorted(cached_directory_paths, key=_get_modification_time)[-1])
    version, hash = cached_directory_path.parts[-2:]
    other_configs = [
        Path(_cached_directory_path).parts[-3]
        for _cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", version, hash))
        if os.path.isdir(_cached_directory_path)
        and (
            config_kwargs
            or custom_features
            or json.loads(Path(_cached_directory_path, "dataset_info.json").read_text(encoding="utf-8"))["config_name"]
            == Path(_cached_directory_path).parts[-3]  # no extra params => config_id == config_name
        )
    ]
    if not config_id and len(other_configs) > 1:
        raise ValueError(
            f"There are multiple '{dataset_name}' configurations in the cache: {', '.join(other_configs)}"
            f"\nPlease specify which configuration to reload from the cache, e.g."
            f"\n\tload_dataset('{dataset_name}', '{other_configs[0]}')"
        )
    config_name = cached_directory_path.parts[-3]
    warning_msg = (
        f"Found the latest cached dataset configuration '{config_name}' at {cached_directory_path} "
        f"(last modified on {time.ctime(_get_modification_time(cached_directory_path))})."
    )
    logger.warning(warning_msg)
    return config_name, version, hash


class Cache(datasets.ArrowBasedBuilder):
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
        version: Optional[str] = "0.0.0",
        hash: Optional[str] = None,
        base_path: Optional[str] = None,
        info: Optional[datasets.DatasetInfo] = None,
        features: Optional[datasets.Features] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token="deprecated",
        repo_id: Optional[str] = None,
        data_files: Optional[Union[str, list, dict, datasets.data_files.DataFilesDict]] = None,
        data_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        writer_batch_size: Optional[int] = None,
        name="deprecated",
        **config_kwargs,
    ):
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'token={use_auth_token}' instead.",
                FutureWarning,
            )
            token = use_auth_token
        if name != "deprecated":
            warnings.warn(
                "Parameter 'name' was renamed to 'config_name' in version 2.3.0 and will be removed in 3.0.0.",
                category=FutureWarning,
            )
            config_name = name
        if repo_id is None and dataset_name is None:
            raise ValueError("repo_id or dataset_name is required for the Cache dataset builder")
        if data_files is not None:
            config_kwargs["data_files"] = data_files
        if data_dir is not None:
            config_kwargs["data_dir"] = data_dir
        if hash == "auto" and version == "auto":
            config_name, version, hash = _find_hash_in_cache(
                dataset_name=repo_id or dataset_name,
                config_name=config_name,
                cache_dir=cache_dir,
                config_kwargs=config_kwargs,
                custom_features=features,
            )
        elif hash == "auto" or version == "auto":
            raise NotImplementedError("Pass both hash='auto' and version='auto' instead")
        super().__init__(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            config_name=config_name,
            version=version,
            hash=hash,
            base_path=base_path,
            info=info,
            token=token,
            repo_id=repo_id,
            storage_options=storage_options,
            writer_batch_size=writer_batch_size,
        )

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo()

    def download_and_prepare(self, output_dir: Optional[str] = None, *args, **kwargs):
        if not os.path.exists(self.cache_dir):
            raise ValueError(f"Cache directory for {self.dataset_name} doesn't exist at {self.cache_dir}")
        if output_dir is not None and output_dir != self.cache_dir:
            shutil.copytree(self.cache_dir, output_dir)

    def _split_generators(self, dl_manager):
        # used to stream from cache
        if isinstance(self.info.splits, datasets.SplitDict):
            split_infos: List[datasets.SplitInfo] = list(self.info.splits.values())
        else:
            raise ValueError(f"Missing splits info for {self.dataset_name} in cache directory {self.cache_dir}")
        return [
            datasets.SplitGenerator(
                name=split_info.name,
                gen_kwargs={
                    "files": filenames_for_dataset_split(
                        self.cache_dir,
                        dataset_name=self.dataset_name,
                        split=split_info.name,
                        filetype_suffix="arrow",
                        shard_lengths=split_info.shard_lengths,
                    )
                },
            )
            for split_info in split_infos
        ]

    def _generate_tables(self, files):
        # used to stream from cache
        for file_idx, file in enumerate(files):
            with open(file, "rb") as f:
                try:
                    for batch_idx, record_batch in enumerate(pa.ipc.open_stream(f)):
                        pa_table = pa.Table.from_batches([record_batch])
                        # Uncomment for debugging (will print the Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                        yield f"{file_idx}_{batch_idx}", pa_table
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise
