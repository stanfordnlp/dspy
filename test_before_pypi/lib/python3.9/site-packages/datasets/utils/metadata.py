import re
import textwrap
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import yaml
from huggingface_hub import DatasetCardData

from ..config import METADATA_CONFIGS_FIELD
from ..info import DatasetInfo, DatasetInfosDict
from ..naming import _split_re
from ..utils.logging import get_logger
from .deprecation_utils import deprecated


logger = get_logger(__name__)


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    def _check_no_duplicates_on_constructed_node(self, node):
        keys = [self.constructed_objects[key_node] for key_node, _ in node.value]
        keys = [tuple(key) if isinstance(key, list) else key for key in keys]
        counter = Counter(keys)
        duplicate_keys = [key for key in counter if counter[key] > 1]
        if duplicate_keys:
            raise TypeError(f"Got duplicate yaml keys: {duplicate_keys}")

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        self._check_no_duplicates_on_constructed_node(node)
        return mapping


def _split_yaml_from_readme(readme_content: str) -> Tuple[Optional[str], str]:
    full_content = list(readme_content.splitlines())
    if full_content and full_content[0] == "---" and "---" in full_content[1:]:
        sep_idx = full_content[1:].index("---") + 1
        yamlblock = "\n".join(full_content[1:sep_idx])
        return yamlblock, "\n".join(full_content[sep_idx + 1 :])

    return None, "\n".join(full_content)


@deprecated("Use `huggingface_hub.DatasetCardData` instead.")
class DatasetMetadata(dict):
    # class attributes
    _FIELDS_WITH_DASHES = {"train_eval_index"}  # train-eval-index in the YAML metadata

    @classmethod
    def from_readme(cls, path: Union[Path, str]) -> "DatasetMetadata":
        """Loads and validates the dataset metadata from its dataset card (README.md)

        Args:
            path (:obj:`Path`): Path to the dataset card (its README.md file)

        Returns:
            :class:`DatasetMetadata`: The dataset's metadata

        Raises:
            :obj:`TypeError`: If the dataset's metadata is invalid
        """
        with open(path, encoding="utf-8") as readme_file:
            yaml_string, _ = _split_yaml_from_readme(readme_file.read())
        if yaml_string is not None:
            return cls.from_yaml_string(yaml_string)
        else:
            return cls()

    def to_readme(self, path: Path):
        if path.exists():
            with open(path, encoding="utf-8") as readme_file:
                readme_content = readme_file.read()
        else:
            readme_content = None
        updated_readme_content = self._to_readme(readme_content)
        with open(path, "w", encoding="utf-8") as readme_file:
            readme_file.write(updated_readme_content)

    def _to_readme(self, readme_content: Optional[str] = None) -> str:
        if readme_content is not None:
            _, content = _split_yaml_from_readme(readme_content)
            full_content = "---\n" + self.to_yaml_string() + "---\n" + content
        else:
            full_content = "---\n" + self.to_yaml_string() + "---\n"
        return full_content

    @classmethod
    def from_yaml_string(cls, string: str) -> "DatasetMetadata":
        """Loads and validates the dataset metadata from a YAML string

        Args:
            string (:obj:`str`): The YAML string

        Returns:
            :class:`DatasetMetadata`: The dataset's metadata

        Raises:
            :obj:`TypeError`: If the dataset's metadata is invalid
        """
        metadata_dict = yaml.load(string, Loader=_NoDuplicateSafeLoader) or {}

        # Convert the YAML keys to DatasetMetadata fields
        metadata_dict = {
            (key.replace("-", "_") if key.replace("-", "_") in cls._FIELDS_WITH_DASHES else key): value
            for key, value in metadata_dict.items()
        }
        return cls(**metadata_dict)

    def to_yaml_string(self) -> str:
        return yaml.safe_dump(
            {
                (key.replace("_", "-") if key in self._FIELDS_WITH_DASHES else key): value
                for key, value in self.items()
            },
            sort_keys=False,
            allow_unicode=True,
            encoding="utf-8",
        ).decode("utf-8")


class MetadataConfigs(Dict[str, Dict[str, Any]]):
    """Should be in format {config_name: {**config_params}}."""

    FIELD_NAME: ClassVar[str] = METADATA_CONFIGS_FIELD

    @staticmethod
    def _raise_if_data_files_field_not_valid(metadata_config: dict):
        yaml_data_files = metadata_config.get("data_files")
        if yaml_data_files is not None:
            yaml_error_message = textwrap.dedent(
                f"""
                Expected data_files in YAML to be either a string or a list of strings
                or a list of dicts with two keys: 'split' and 'path', but got {yaml_data_files}
                Examples of data_files in YAML:

                   data_files: data.csv

                   data_files: data/*.png

                   data_files:
                    - part0/*
                    - part1/*

                   data_files:
                    - split: train
                      path: train/*
                    - split: test
                      path: test/*

                   data_files:
                    - split: train
                      path:
                      - train/part1/*
                      - train/part2/*
                    - split: test
                      path: test/*

                PS: some symbols like dashes '-' are not allowed in split names
                """
            )
            if not isinstance(yaml_data_files, (list, str)):
                raise ValueError(yaml_error_message)
            if isinstance(yaml_data_files, list):
                for yaml_data_files_item in yaml_data_files:
                    if (
                        not isinstance(yaml_data_files_item, (str, dict))
                        or isinstance(yaml_data_files_item, dict)
                        and not (
                            len(yaml_data_files_item) == 2
                            and "split" in yaml_data_files_item
                            and re.match(_split_re, yaml_data_files_item["split"])
                            and isinstance(yaml_data_files_item.get("path"), (str, list))
                        )
                    ):
                        raise ValueError(yaml_error_message)

    @classmethod
    def _from_exported_parquet_files_and_dataset_infos(
        cls,
        revision: str,
        exported_parquet_files: List[Dict[str, Any]],
        dataset_infos: DatasetInfosDict,
    ) -> "MetadataConfigs":
        metadata_configs = {
            config_name: {
                "data_files": [
                    {
                        "split": split_name,
                        "path": [
                            parquet_file["url"].replace("refs%2Fconvert%2Fparquet", revision)
                            for parquet_file in parquet_files_for_split
                        ],
                    }
                    for split_name, parquet_files_for_split in groupby(parquet_files_for_config, itemgetter("split"))
                ],
                "version": str(dataset_infos.get(config_name, DatasetInfo()).version or "0.0.0"),
            }
            for config_name, parquet_files_for_config in groupby(exported_parquet_files, itemgetter("config"))
        }
        if dataset_infos:
            # Preserve order of configs and splits
            metadata_configs = {
                config_name: {
                    "data_files": [
                        data_file
                        for split_name in dataset_info.splits
                        for data_file in metadata_configs[config_name]["data_files"]
                        if data_file["split"] == split_name
                    ],
                    "version": metadata_configs[config_name]["version"],
                }
                for config_name, dataset_info in dataset_infos.items()
            }
        return cls(metadata_configs)

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> "MetadataConfigs":
        if dataset_card_data.get(cls.FIELD_NAME):
            metadata_configs = dataset_card_data[cls.FIELD_NAME]
            if not isinstance(metadata_configs, list):
                raise ValueError(f"Expected {cls.FIELD_NAME} to be a list, but got '{metadata_configs}'")
            for metadata_config in metadata_configs:
                if "config_name" not in metadata_config:
                    raise ValueError(
                        f"Each config must include `config_name` field with a string name of a config, "
                        f"but got {metadata_config}. "
                    )
                cls._raise_if_data_files_field_not_valid(metadata_config)
            return cls(
                {
                    config["config_name"]: {param: value for param, value in config.items() if param != "config_name"}
                    for config in metadata_configs
                }
            )
        return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            for metadata_config in self.values():
                self._raise_if_data_files_field_not_valid(metadata_config)
            current_metadata_configs = self.from_dataset_card_data(dataset_card_data)
            total_metadata_configs = dict(sorted({**current_metadata_configs, **self}.items()))
            for config_name, config_metadata in total_metadata_configs.items():
                config_metadata.pop("config_name", None)
            dataset_card_data[self.FIELD_NAME] = [
                {"config_name": config_name, **config_metadata}
                for config_name, config_metadata in total_metadata_configs.items()
            ]

    def get_default_config_name(self) -> Optional[str]:
        default_config_name = None
        for config_name, metadata_config in self.items():
            if len(self) == 1 or config_name == "default" or metadata_config.get("default"):
                if default_config_name is None:
                    default_config_name = config_name
                else:
                    raise ValueError(
                        f"Dataset has several default configs: '{default_config_name}' and '{config_name}'."
                    )
        return default_config_name


# DEPRECATED - just here to support old versions of evaluate like 0.2.2
# To support new tasks on the Hugging Face Hub, please open a PR for this file:
# https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/pipelines.ts
known_task_ids = {
    "image-classification": [],
    "translation": [],
    "image-segmentation": [],
    "fill-mask": [],
    "automatic-speech-recognition": [],
    "token-classification": [],
    "sentence-similarity": [],
    "audio-classification": [],
    "question-answering": [],
    "summarization": [],
    "zero-shot-classification": [],
    "table-to-text": [],
    "feature-extraction": [],
    "other": [],
    "multiple-choice": [],
    "text-classification": [],
    "text-to-image": [],
    "text2text-generation": [],
    "zero-shot-image-classification": [],
    "tabular-classification": [],
    "tabular-regression": [],
    "image-to-image": [],
    "tabular-to-text": [],
    "unconditional-image-generation": [],
    "text-retrieval": [],
    "text-to-speech": [],
    "object-detection": [],
    "audio-to-audio": [],
    "text-generation": [],
    "conversational": [],
    "table-question-answering": [],
    "visual-question-answering": [],
    "image-to-text": [],
    "reinforcement-learning": [],
    "voice-activity-detection": [],
    "time-series-forecasting": [],
    "document-question-answering": [],
}


if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser(usage="Validate the yaml metadata block of a README.md file.")
    ap.add_argument("readme_filepath")
    args = ap.parse_args()

    readme_filepath = Path(args.readme_filepath)
    dataset_metadata = DatasetMetadata.from_readme(readme_filepath)
    print(dataset_metadata)
    dataset_metadata.to_readme(readme_filepath)
