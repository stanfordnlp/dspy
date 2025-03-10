import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests
import ujson

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, get_finetune_directory

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


class TrainingJobDatabricks(TrainingJob):
    def __init__(self, finetuning_run=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetuning_run = finetuning_run
        self.launch_started = False
        self.launch_completed = False
        self.endpoint_name = None

    def status(self):
        if not self.finetuning_run:
            return None
        try:
            from databricks.model_training import foundation_model as fm
        except ImportError:
            raise ImportError(
                "To use Databricks finetuning, please install the databricks_genai package via "
                "`pip install databricks_genai`."
            )
        run = fm.get(self.finetuning_run)
        return run.status


class DatabricksProvider(Provider):
    finetunable = True
    TrainingJob = TrainingJobDatabricks

    @staticmethod
    def is_provider_model(model: str) -> bool:
        # We don't automatically infer Databricks models because Databricks is not a proprietary model provider.
        return False

    @staticmethod
    def deploy_finetuned_model(
        model: str,
        data_format: Optional[TrainDataFormat] = None,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        deploy_timeout: int = 900,
    ):
        workspace_client = _get_workspace_client()
        model_version = next(workspace_client.model_versions.list(model)).version

        # Allow users to override the host and token. This is useful on Databricks hosted runtime.
        databricks_host = databricks_host or workspace_client.config.host
        databricks_token = databricks_token or workspace_client.config.token

        headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

        optimizable_info = requests.get(
            url=f"{databricks_host}/api/2.0/serving-endpoints/get-model-optimization-info/{model}/{model_version}",
            headers=headers,
        ).json()

        if "optimizable" not in optimizable_info or not optimizable_info["optimizable"]:
            raise ValueError(f"Model is not eligible for provisioned throughput: {optimizable_info}")

        chunk_size = optimizable_info["throughput_chunk_size"]

        # Minimum desired provisioned throughput
        min_provisioned_throughput = 0

        # Maximum desired provisioned throughput
        max_provisioned_throughput = chunk_size

        # Databricks serving endpoint names cannot contain ".".
        model_name = model.replace(".", "_")

        get_endpoint_response = requests.get(
            url=f"{databricks_host}/api/2.0/serving-endpoints/{model_name}", json={"name": model_name}, headers=headers
        )

        if get_endpoint_response.status_code == 200:
            logger.info(f"Serving endpoint {model_name} already exists, updating it instead of creating a new one.")
            # The serving endpoint already exists, we will update it instead of creating a new one.
            data = {
                "served_entities": [
                    {
                        "name": model_name,
                        "entity_name": model,
                        "entity_version": model_version,
                        "min_provisioned_throughput": min_provisioned_throughput,
                        "max_provisioned_throughput": max_provisioned_throughput,
                    }
                ]
            }

            response = requests.put(
                url=f"{databricks_host}/api/2.0/serving-endpoints/{model_name}/config",
                json=data,
                headers=headers,
            )
        else:
            logger.info(f"Creating serving endpoint {model_name} on Databricks model serving!")
            # Send the POST request to create the serving endpoint.
            data = {
                "name": model_name,
                "config": {
                    "served_entities": [
                        {
                            "name": model_name,
                            "entity_name": model,
                            "entity_version": model_version,
                            "min_provisioned_throughput": min_provisioned_throughput,
                            "max_provisioned_throughput": max_provisioned_throughput,
                        }
                    ]
                },
            }

            response = requests.post(url=f"{databricks_host}/api/2.0/serving-endpoints", json=data, headers=headers)

        if response.status_code == 200:
            logger.info(
                f"Successfully started creating/updating serving endpoint {model_name} on Databricks model serving!"
            )
        else:
            raise ValueError(f"Failed to create serving endpoint: {response.json()}.")

        logger.info(
            f"Waiting for serving endpoint {model_name} to be ready, this might take a few minutes... You can check "
            f"the status of the endpoint at {databricks_host}/ml/endpoints/{model_name}"
        )
        from openai import OpenAI

        client = OpenAI(
            api_key=databricks_token,
            base_url=f"{databricks_host}/serving-endpoints",
        )
        # Wait for the deployment to be ready.
        num_retries = deploy_timeout // 60
        for _ in range(num_retries):
            try:
                if data_format == TrainDataFormat.CHAT:
                    client.chat.completions.create(
                        messages=[{"role": "user", "content": "hi"}], model=model_name, max_tokens=1
                    )
                elif data_format == TrainDataFormat.COMPLETION:
                    client.completions.create(prompt="hi", model=model_name, max_tokens=1)
                logger.info(f"Databricks model serving endpoint {model_name} is ready!")
                return
            except Exception:
                time.sleep(60)

        raise ValueError(
            f"Failed to create serving endpoint {model_name} on Databricks model serving platform within "
            f"{deploy_timeout} seconds."
        )

    @staticmethod
    def finetune(
        job: TrainingJobDatabricks,
        model: str,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[Union[TrainDataFormat, str]] = "chat",
        train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if isinstance(train_data_format, str):
            if train_data_format == "chat":
                train_data_format = TrainDataFormat.CHAT
            elif train_data_format == "completion":
                train_data_format = TrainDataFormat.COMPLETION
            else:
                raise ValueError(
                    f"String `train_data_format` must be one of 'chat' or 'completion', but received: {train_data_format}."
                )

        if "train_data_path" not in train_kwargs:
            raise ValueError("The `train_data_path` must be provided to finetune on Databricks.")
        # Add the file name to the directory path.
        train_kwargs["train_data_path"] = DatabricksProvider.upload_data(
            train_data, train_kwargs["train_data_path"], train_data_format
        )

        try:
            from databricks.model_training import foundation_model as fm
        except ImportError:
            raise ImportError(
                "To use Databricks finetuning, please install the databricks_genai package via "
                "`pip install databricks_genai`."
            )

        if "register_to" not in train_kwargs:
            raise ValueError("The `register_to` must be provided to finetune on Databricks.")

        # Allow users to override the host and token. This is useful on Databricks hosted runtime.
        databricks_host = train_kwargs.pop("databricks_host", None)
        databricks_token = train_kwargs.pop("databricks_token", None)

        skip_deploy = train_kwargs.pop("skip_deploy", False)
        deploy_timeout = train_kwargs.pop("deploy_timeout", 900)

        logger.info("Starting finetuning on Databricks... this might take a few minutes to finish.")
        finetuning_run = fm.create(
            model=model,
            **train_kwargs,
        )

        job.run = finetuning_run

        # Wait for the finetuning run to be ready.
        while True:
            job.run = fm.get(job.run)
            if job.run.status.display_name == "Completed":
                logger.info("Finetuning run completed successfully!")
                break
            elif job.run.status.display_name == "Failed":
                raise ValueError(
                    f"Finetuning run failed with status: {job.run.status.display_name}. Please check the Databricks "
                    f"workspace for more details. Finetuning job's metadata: {job.run}."
                )
            else:
                time.sleep(60)

        if skip_deploy:
            return None

        job.launch_started = True
        model_to_deploy = train_kwargs.get("register_to")
        job.endpoint_name = model_to_deploy.replace(".", "_")
        DatabricksProvider.deploy_finetuned_model(
            model_to_deploy, train_data_format, databricks_host, databricks_token, deploy_timeout
        )
        job.launch_completed = True
        # The finetuned model name should be in the format: "databricks/<endpoint_name>".
        return f"databricks/{job.endpoint_name}"

    @staticmethod
    def upload_data(train_data: List[Dict[str, Any]], databricks_unity_catalog_path: str, data_format: TrainDataFormat):
        logger.info("Uploading finetuning data to Databricks Unity Catalog...")
        file_path = _save_data_to_local_file(train_data, data_format)

        w = _get_workspace_client()
        _create_directory_in_databricks_unity_catalog(w, databricks_unity_catalog_path)

        try:
            with open(file_path, "rb") as f:
                target_path = os.path.join(databricks_unity_catalog_path, os.path.basename(file_path))
                w.files.upload(target_path, f, overwrite=True)
            logger.info("Successfully uploaded finetuning data to Databricks Unity Catalog!")
            return target_path
        except Exception as e:
            raise ValueError(f"Failed to upload finetuning data to Databricks Unity Catalog: {e}")


def _get_workspace_client() -> "WorkspaceClient":
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "To use Databricks finetuning, please install the databricks-sdk package via "
            "`pip install databricks-sdk`."
        )
    return WorkspaceClient()


def _create_directory_in_databricks_unity_catalog(w: "WorkspaceClient", databricks_unity_catalog_path: str):
    pattern = r"^/Volumes/(?P<catalog>[^/]+)/(?P<schema>[^/]+)/(?P<volume>[^/]+)(/[^/]+)+$"
    match = re.match(pattern, databricks_unity_catalog_path)
    if not match:
        raise ValueError(
            f"Databricks Unity Catalog path must be in the format '/Volumes/<catalog>/<schema>/<volume>/...', but "
            f"received: {databricks_unity_catalog_path}."
        )

    catalog = match.group("catalog")
    schema = match.group("schema")
    volume = match.group("volume")

    try:
        volume_path = f"{catalog}.{schema}.{volume}"
        w.volumes.read(volume_path)
    except Exception:
        raise ValueError(
            f"Databricks Unity Catalog volume does not exist: {volume_path}, please create it on the Databricks "
            "workspace."
        )

    try:
        w.files.get_directory_metadata(databricks_unity_catalog_path)
        logger.info(f"Directory {databricks_unity_catalog_path} already exists, skip creating it.")
    except Exception:
        # Create the directory if it doesn't exist, we don't raise an error because this is a common case.
        logger.info(f"Creating directory {databricks_unity_catalog_path} in Databricks Unity Catalog...")
        w.files.create_directory(databricks_unity_catalog_path)
        logger.info(f"Successfully created directory {databricks_unity_catalog_path} in Databricks Unity Catalog!")


def _save_data_to_local_file(train_data: List[Dict[str, Any]], data_format: TrainDataFormat):
    import uuid

    file_name = f"finetuning_{uuid.uuid4()}.jsonl"

    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "w") as f:
        for item in train_data:
            if data_format == TrainDataFormat.CHAT:
                _validate_chat_data(item)
            elif data_format == TrainDataFormat.COMPLETION:
                _validate_completion_data(item)

            f.write(ujson.dumps(item) + "\n")
    return file_path


def _validate_chat_data(data: Dict[str, Any]):
    if "messages" not in data:
        raise ValueError(
            "Each finetuning data must be a dict with a 'messages' key when `task=CHAT_COMPLETION`, but "
            f"received: {data}"
        )

    if not isinstance(data["messages"], list):
        raise ValueError(
            "The value of the 'messages' key in each finetuning data must be a list of dicts with keys 'role' and "
            f"'content' when `task=CHAT_COMPLETION`, but received: {data['messages']}"
        )

    for message in data["messages"]:
        if "role" not in message:
            raise ValueError(f"Each message in the 'messages' list must contain a 'role' key, but received: {message}.")
        if "content" not in message:
            raise ValueError(
                f"Each message in the 'messages' list must contain a 'content' key, but received: {message}."
            )


def _validate_completion_data(data: Dict[str, Any]):
    if "prompt" not in data:
        raise ValueError(
            "Each finetuning data must be a dict with a 'prompt' key when `task=INSTRUCTION_FINETUNE`, but "
            f"received: {data}"
        )
    if "response" not in data and "completion" not in data:
        raise ValueError(
            "Each finetuning data must be a dict with a 'response' or 'completion' key when "
            f"`task=INSTRUCTION_FINETUNE`, but received: {data}"
        )
