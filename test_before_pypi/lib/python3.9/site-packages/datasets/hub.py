import time
from itertools import chain
from typing import Optional, Union

from huggingface_hub import (
    CommitInfo,
    CommitOperationAdd,
    CommitOperationDelete,
    DatasetCard,
    DatasetCardData,
    HfApi,
    HfFileSystem,
)
from huggingface_hub.utils import HfHubHTTPError

import datasets.config
from datasets.info import DatasetInfosDict
from datasets.inspect import get_dataset_config_names, get_dataset_default_config_name
from datasets.load import load_dataset, load_dataset_builder
from datasets.utils.metadata import MetadataConfigs


def convert_to_parquet(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    trust_remote_code: Optional[bool] = None,
) -> CommitInfo:
    """Convert Hub [script-based dataset](dataset_script) to Parquet [data-only dataset](repository_structure), so that
    the dataset viewer will be supported.

    This function:
    - makes a copy of the script on the "main" branch into a dedicated branch called "script" (if it does not already exist)
    - creates a pull request to the Hub dataset to convert it to Parquet files (and deletes the script from the main branch)

    If in the future you need to recreate the Parquet files from the "script" branch, pass the `revision="script"` argument.

    Note that you should pass the `trust_remote_code=True` argument only if you trust the remote code to be executed locally on your machine.

    Args:
        repo_id (`str`): ID of the source Hub dataset repository, in the following format: `<user>/<dataset_name>` or
            `<org>/<dataset_name>`.
        revision (`str`, *optional*): Branch of the source Hub dataset repository. Defaults to the `"main"` branch.
        token (`bool` or `str`, *optional*): Authentication token for the Hugging Face Hub.
        trust_remote_code (`bool`, defaults to `False`): Whether you trust the remote code of the Hub script-based
            dataset to be executed locally on your machine. This option should only be set to `True` for repositories
            where you have read the code and which you trust.

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

    Returns:
        `huggingface_hub.CommitInfo`
    """
    print(f"{repo_id}")
    configs = get_dataset_config_names(repo_id, token=token, revision=revision, trust_remote_code=trust_remote_code)
    print(f"{configs = }")
    default_config = get_dataset_default_config_name(
        repo_id, token=token, revision=revision, trust_remote_code=trust_remote_code
    )
    print(f"{default_config = }")
    if default_config:
        config = default_config
        configs.remove(default_config)
    else:
        config = configs.pop(0)
    print(f"{config = }")
    dataset = load_dataset(repo_id, config, revision=revision, trust_remote_code=trust_remote_code)
    commit_info = dataset.push_to_hub(
        repo_id,
        config_name=config,
        commit_message="Convert dataset to Parquet",
        commit_description="Convert dataset to Parquet.",
        create_pr=True,
        token=token,
        set_default=default_config is not None,
    )
    time.sleep(5)
    pr_revision, pr_url = commit_info.pr_revision, commit_info.pr_url
    for config in configs:
        print(f"{config = }")
        dataset = load_dataset(repo_id, config, revision=revision, trust_remote_code=trust_remote_code)
        dataset.push_to_hub(
            repo_id,
            config_name=config,
            commit_message=f"Add '{config}' config data files",
            revision=pr_revision,
            token=token,
        )
        time.sleep(5)
    _delete_files(repo_id, revision=pr_revision, token=token)
    if not revision:
        api = HfApi(endpoint=datasets.config.HF_ENDPOINT, token=token)
        try:
            api.create_branch(repo_id, branch="script", repo_type="dataset", token=token, exist_ok=True)
        except HfHubHTTPError:
            pass
    print(f"You can find your PR to convert the dataset to Parquet at: {pr_url}")
    return commit_info


def delete_from_hub(
    repo_id: str,
    config_name: str,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
) -> CommitInfo:
    """Delete a dataset configuration from a [data-only dataset](repository_structure) on the Hub.

    Args:
        repo_id (`str`): ID of the Hub dataset repository, in the following format: `<user>/<dataset_name>` or
            `<org>/<dataset_name>`.
        config_name (`str`): Name of the dataset configuration.
        revision (`str`, *optional*): Branch to delete the configuration from. Defaults to the `"main"` branch.
        token (`bool` or `str`, *optional*): Authentication token for the Hugging Face Hub.

    Returns:
        `huggingface_hub.CommitInfo`
    """
    operations = []
    # data_files
    fs = HfFileSystem(endpoint=datasets.config.HF_ENDPOINT, token=token)
    builder = load_dataset_builder(repo_id, config_name, revision=revision, token=token, trust_remote_code=False)
    for data_file in chain(*builder.config.data_files.values()):
        data_file_resolved_path = fs.resolve_path(data_file)
        if data_file_resolved_path.repo_id == repo_id:
            operations.append(CommitOperationDelete(path_in_repo=data_file_resolved_path.path_in_repo))
    # README.md
    dataset_card = DatasetCard.load(repo_id)
    # config_names
    if dataset_card.data.get("config_names", None) and config_name in dataset_card.data["config_names"]:
        dataset_card.data["config_names"].remove(config_name)
    # metadata_configs
    metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card.data)
    if metadata_configs:
        _ = metadata_configs.pop(config_name, None)
        dataset_card_data = DatasetCardData()
        metadata_configs.to_dataset_card_data(dataset_card_data)
        if datasets.config.METADATA_CONFIGS_FIELD in dataset_card_data:
            dataset_card.data[datasets.config.METADATA_CONFIGS_FIELD] = dataset_card_data[
                datasets.config.METADATA_CONFIGS_FIELD
            ]
        else:
            _ = dataset_card.data.pop(datasets.config.METADATA_CONFIGS_FIELD, None)
    # dataset_info
    dataset_infos: DatasetInfosDict = DatasetInfosDict.from_dataset_card_data(dataset_card.data)
    if dataset_infos:
        _ = dataset_infos.pop(config_name, None)
        dataset_card_data = DatasetCardData()
        dataset_infos.to_dataset_card_data(dataset_card_data)
        if "dataset_info" in dataset_card_data:
            dataset_card.data["dataset_info"] = dataset_card_data["dataset_info"]
        else:
            _ = dataset_card.data.pop("dataset_info", None)
    # Commit
    operations.append(
        CommitOperationAdd(path_in_repo=datasets.config.REPOCARD_FILENAME, path_or_fileobj=str(dataset_card).encode())
    )
    api = HfApi(endpoint=datasets.config.HF_ENDPOINT, token=token)
    commit_info = api.create_commit(
        repo_id,
        operations=operations,
        commit_message=f"Delete '{config_name}' config",
        commit_description=f"Delete '{config_name}' config.",
        token=token,
        repo_type="dataset",
        revision=revision,
        create_pr=True,
    )
    print(f"You can find your PR to delete the dataset config at: {commit_info.pr_url}")
    return commit_info


def _delete_files(dataset_id, revision=None, token=None):
    dataset_name = dataset_id.split("/")[-1]
    hf_api = HfApi(endpoint=datasets.config.HF_ENDPOINT, token=token)
    repo_files = hf_api.list_repo_files(
        dataset_id,
        repo_type="dataset",
    )
    if repo_files:
        legacy_json_file = []
        python_files = []
        data_files = []
        for filename in repo_files:
            if filename in {".gitattributes", "README.md"}:
                continue
            elif filename == f"{dataset_name}.py":
                hf_api.delete_file(
                    filename,
                    dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    commit_message="Delete loading script",
                )
            elif filename == "dataset_infos.json":
                legacy_json_file.append(filename)
            elif filename.endswith(".py"):
                python_files.append(filename)
            else:
                data_files.append(filename)
        if legacy_json_file:
            hf_api.delete_file(
                "dataset_infos.json",
                dataset_id,
                repo_type="dataset",
                revision=revision,
                commit_message="Delete legacy dataset_infos.json",
            )
        if python_files:
            for filename in python_files:
                hf_api.delete_file(
                    filename,
                    dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    commit_message="Delete loading script auxiliary file",
                )
        if data_files:
            for filename in data_files:
                hf_api.delete_file(
                    filename,
                    dataset_id,
                    repo_type="dataset",
                    revision=revision,
                    commit_message="Delete data file",
                )
