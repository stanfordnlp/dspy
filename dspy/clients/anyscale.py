from typing import Any, Dict, List, Optional, Union
import ujson
import yaml
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import functools

from dspy import logger
from dspy.clients.finetune import (
    FinetuneJob,
    TrainingMethod
)
import asyncio

try:
    from anyscale.job import JobConfig
    import anyscale
except ImportError:
    anyscale = None

#-------------------------------------------------------------------------------
#    Variables
#-------------------------------------------------------------------------------

# List of training methods supported by AnyScale
TRAINING_METHODS_ANYSCALE = [
    TrainingMethod.SFT,
]

#-------------------------------------------------------------------------------
#    Launching and killing LMs
#-------------------------------------------------------------------------------

def anyscale_model_launch(model: str, launch_kwargs: Dict[str, Any]):
    """Launch an AnyScale model."""
    # TODO: Hardcode resources for launching a select server through docker
    raise NotImplementedError("Method `anyscale_model_launch` is not implemented.")


def anyscale_model_kill(model: str, launch_kwargs: Dict[str, Any]):
    # TODO: Hardcode resources for killing a select server through docker
    """Kill an AnyScale model."""
    raise NotImplementedError("Method `anyscale_model_kill` is not implemented.")


#-------------------------------------------------------------------------------
#    Function and classes required for the fine-tune interface
#-------------------------------------------------------------------------------

class FinetuneJobAnyScale(FinetuneJob):
    def __init__(self,
        model: str,
        # message_completion_pairs: List[Dict[str, str]],
        train_path: str,
        eval_path: Optional[str],
        train_kwargs: Optional[Dict[str, Any]]=None,
    ):
        super().__init__(model, train_path, eval_path, train_kwargs)
        self.model = model
        # self.message_completion_pairs = message_completion_pairs
        self.train_path = train_path
        self.train_kwargs: Dict[str, Any] = train_kwargs or {}
        if "model" not in self.train_kwargs:
            self.train_kwargs["model"] = model
        self.job_id = None

    def cancel(self):
        """Cancel the finetune job."""
        raise NotImplementedError("Method `cancel` is not implemented.")
        # Call the super's cancel method after the custom cancellation logic, 
        # so that the future can be cancelled
        # super().cancel()

    def status(self):
        """Get the status of the finetune job."""
        raise NotImplementedError("Method `status` is not implemented.")

    async def run_finetune(self):
        """Start the finetune job."""
        try:
            logger.info("[Finetune] Starting training process...")
            if TrainingMethod.SFT not in TRAINING_METHODS_ANYSCALE:
                raise NotImplementedError(f"AnyScale can only support {TRAINING_METHODS_ANYSCALE} for the time being")

            logger.info("[Finetune] Reading and validating datasets...")
            # Convert the data from prompt completion to OAI compatible messages
            train_dataset = read_jsonl(self.train_path)
            val_dataset = read_jsonl(self.eval_path) if self.eval_path else None

            if not verify_datasets(train_dataset, val_dataset):
                logger.error("[Finetune] Error: Unable to verify arguments")
                raise RuntimeError("Unable to verify argument")

            logger.info("[Finetune] Converting data to JSONL format...")
            for path, dataset in [(self.train_path, train_dataset), (self.eval_path, val_dataset)]:
                if not (path and dataset):
                    continue
                write_jsonl(path, dataset)

            logger.info("[Finetune] Submitting data to remote storage...")
            remote_train_path, remote_eval_path = submit_data(train_path=self.train_path, eval_path=self.eval_path)
            logger.info(f"[Finetune] Data submitted. Remote train path: {remote_train_path}, Remote eval path: {remote_eval_path}")

            logger.info("[Finetune] Generating configuration files...")
            _, compute_config = generate_config_files(train_path=remote_train_path, eval_path=remote_eval_path, **self.train_kwargs)
            
            logger.info("[Finetune] Starting remote training...")
            job_id = start_remote_training(compute_config=compute_config, **self.train_kwargs)
            self.job_id = job_id
            logger.info(f"[Finetune] Remote training started. Job ID: {job_id}")

            logger.info("[Finetune] Waiting for training to complete...")
            await self.wait_for_training()
            logger.info("[Finetune] Training completed.")

            logger.info("[Finetune] Retrieving model information...")
            model_info = get_model_info(job_id)
            logger.info(f"[Finetune] Model info retrieved: {model_info}")

            storage_uri = model_info["storage_uri"]
            logger.info(f"[Finetune] Copying LoRA weights from {storage_uri}...")
            model_names = copy_lora_weights(storage_uri, model_info, job_id)
            logger.info(f"[Finetune] LoRA weights copied. Model names: {model_names}")

            logger.info("[Finetune] Setting result in future object...")
            for model_name in model_names:
                yield model_name
            logger.info("[Finetune] Training process completed successfully.")

        except Exception as e:
            logger.error(f"[Finetune] Error occurred during training: {str(e)}")
            raise e

        # executor = ThreadPoolExecutor(max_workers=1)
        # executor.submit(
        #     execute_finetune_job,
        #     finetune_job,
        #     launch_kwargs=launch_kwargs,
        #     cache_finetune=cache_finetune
        # )
        # executor.shutdown(wait=False)
    
    async def wait_for_training(self):
        """Wait for the training to complete in a non-blocking manner."""
        loop = asyncio.get_running_loop()
        
        # Run the blocking `anyscale.job.wait` in a separate thread
        
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, functools.partial(anyscale.job.wait, id=self.job_id, timeout_s=18000))

    


def is_anyscale_model(model: str) -> bool:
    """Check if the model is an AnyScale model."""
    logger.info("Is AnyScale model is not implemented, returning False as a default to not break lm.py")
    return False

# async def _run_finetune(job: FinetuneJobAnyScale):
#         try:
#             logger.info("[Finetune] Starting training process...")
#             if TrainingMethod.SFT not in TRAINING_METHODS_ANYSCALE:
#                 raise NotImplementedError(f"AnyScale can only support {TRAINING_METHODS_ANYSCALE} for the time being")

#             logger.info("[Finetune] Reading and validating datasets...")
#             # Convert the data from prompt completion to OAI compatible messages
#             train_dataset = read_jsonl(train_path)
#             val_dataset = read_jsonl(eval_path) if eval_path else None

#             if not verify_datasets(train_dataset, val_dataset):
#                 logger.error("[Finetune] Error: Unable to verify arguments")
#                 raise RuntimeError("Unable to verify argument")

#             logger.info("[Finetune] Converting data to JSONL format...")
#             for path, dataset in [(train_path, train_dataset), (eval_path, val_dataset)]:
#                 if not (path and dataset):
#                     continue
#                 write_jsonl(path, dataset)

#             logger.info("[Finetune] Submitting data to remote storage...")
#             remote_train_path, remote_eval_path = submit_data(train_path=train_path, eval_path=eval_path)
#             logger.info(f"[Finetune] Data submitted. Remote train path: {remote_train_path}, Remote eval path: {remote_eval_path}")

#             logger.info("[Finetune] Generating configuration files...")
#             _, compute_config = generate_config_files(train_path=remote_train_path, eval_path=remote_eval_path, **train_kwargs)
            
#             logger.info("[Finetune] Starting remote training...")
#             job_id = start_remote_training(compute_config=compute_config, **train_kwargs)
#             job.job_id = job_id
#             logger.info(f"[Finetune] Remote training started. Job ID: {job_id}")

#             logger.info("[Finetune] Waiting for training to complete...")
#             wait_for_training(job)
#             logger.info("[Finetune] Training completed.")

#             logger.info("[Finetune] Retrieving model information...")
#             model_info = get_model_info(job_id)
#             logger.info(f"[Finetune] Model info retrieved: {model_info}")

#             storage_uri = model_info["storage_uri"]
#             logger.info(f"[Finetune] Copying LoRA weights from {storage_uri}...")
#             model_names = copy_lora_weights(storage_uri, model_info, job_id)
#             logger.info(f"[Finetune] LoRA weights copied. Model names: {model_names}")

#             logger.info("[Finetune] Setting result in future object...")
#             job.set_result(model_names)
#             logger.info("[Finetune] Training process completed successfully.")

#         except Exception as e:
#             logger.error(f"[Finetune] Error occurred during training: {str(e)}")
#             job.set_exception(e)


# async def finetune_anyscale(
#     model: str,
#     train_path: str,
#     eval_path: Optional[str],
#     train_kwargs: Optional[Dict[str, Any]] = None,
# ) -> FinetuneJobAnyScale:
#     """
#     Initiate fine-tuning process for an Anyscale model.
    
#     Args:
#         model (str): The name of the Anyscale model to fine-tune.
#         message_completion_pairs (List[Dict[str, str]]): List of prompt-completion pairs for training.
#         train_kwargs (Optional[Dict[str, Any]]): Additional training parameters.
    
#     Returns:
#         FinetuneJobAnyScale: A job object representing the fine-tuning process.
#     """
#     train_kwargs = train_kwargs or {}
    
    

#     executor = ThreadPoolExecutor(max_workers=1)
#     executor.submit(_run_finetune)
    
#     return job


def verify_datasets(dataset: List[dict[str, Any]], valset: Optional[List[dict[str, Any]]]) -> bool:
    """Verify the training arguments before starting training."""
    def validate_dataset(name, data: dict[str, Any]) -> bool:
        dataset_validation = openai_data_validation(data)

        if dataset_validation:
            logger.error(f"Dataset validation failed: {dataset_validation}")
            return False

        if name == "train":
            convo_lens = check_message_lengths(data)
            estimate_cost(data, convo_lens=convo_lens)
        return True

    datasets = {"train": dataset}
    if valset:
        datasets["val"] = valset

    for name, data in datasets.items():
        if not validate_dataset(name, data):
            return False
    return True

def submit_data(train_path: str, eval_path: Optional[str]):
    """Upload the data to the Workspace cloud storage."""
    storage = os.environ['ANYSCALE_ARTIFACT_STORAGE']
    
    datasets = {"train": train_path}
    if eval_path:
        datasets["val"] = eval_path

    fine_tuning_file_ids = {}
    for name, path in datasets.items():
        num_items = len(read_jsonl(path))
        logger.info(f"Number of items in {name} data: {num_items}")

        remote_path = os.path.join(storage, path.split("/")[-1])
        logger.info(f"Uploading {name} data to S3 at {remote_path}")
        if remote_path[:2] == "s3":
            os.system(f"aws s3 cp {path} {remote_path}")
        elif remote_path[:2] == "gs":
            os.system(f"gcloud storage cp {path} {remote_path}")
        else:
            os.system(f"cp {path} {remote_path}")
            logger.info(f"Copied {path} to {remote_path}")
        fine_tuning_file_ids[name] = remote_path
    
    return fine_tuning_file_ids["train"], fine_tuning_file_ids.get("val", None)

def generate_config_files(train_path: str, eval_path: Optional[str], **kwargs):
    base_model_yaml_path = kwargs.get("train_config_yaml", None)
    assert kwargs["model"] is not None, "Model is required to generate the config files"

    use_lora = kwargs.get("use_lora", False)
    example_dir = ""
    lora_path = "configs/training/lora" if use_lora else "configs/training/full_param"


    if not base_model_yaml_path:
        def get_yaml_config(model_name):
            if "llama" in model_name.lower():
                if "70b" in model_name:
                    return "llama-3-70b.yaml"
                elif "13b" in model_name:
                    return "llama-3-70b.yaml"
                else:
                    return "llama-3-8b.yaml"
            elif "mistral" in model_name.lower():
                if "mixtral" in model_name.lower():
                    return "mixtral-8x7b.yaml"
                else:
                    return "mistral-7b.yaml"
            else:
                raise RuntimeError("No default yaml found for the model")

        default_model_yaml_path = get_yaml_config(kwargs["model"])
        base_model_yaml_path = os.path.join(example_dir, lora_path, default_model_yaml_path)
        logger.info(f"Using default yaml template for model: {base_model_yaml_path}")
        
    model_config_data = yaml.safe_load(open(base_model_yaml_path, "r"))
    model_config_data.update(kwargs.get("hyperparameters", {}))
    
    model_config_data["model_id"] = kwargs["model"]

    custom_modifications = {
        "model_id": kwargs["model"],
        "train_path": train_path,
        "valid_path": eval_path,
        "logger": {
            "provider": "wandb",
        },
        "num_checkpoints_to_keep": 10
    }
    if kwargs.get("output_dir", None):
        custom_modifications["output_dir"] = kwargs["output_dir"]

    model_config_data.update(custom_modifications)
    model_config_data = {k: v for k, v in model_config_data.items() if v is not None}

    def freeze(d):
        if isinstance(d, dict):
            return tuple(sorted((key, freeze(value)) for key, value in d.items()))
        elif isinstance(d, list):
            return tuple(freeze(value) for value in sorted(d))
        elif isinstance(d, set):
            return tuple(freeze(value) for value in sorted(d))
        return d

    def hash_dict(d):
        return hash(freeze(d))
    dict_sorted_hash = hash_dict(model_config_data)
    if dict_sorted_hash < 0:
        dict_sorted_hash = -dict_sorted_hash
    filename = f"model_config_dspy_{dict_sorted_hash}.yaml"
    logger.info(f"Model config data: {model_config_data}")
    yaml.safe_dump(model_config_data, open(filename, "w"))

    ft_path = os.path.join("utils", "ft.py")

    compute_config_dict = {
        "name": "dspy-llmforge-fine-tuning-job",
        "entrypoint": f"llmforge anyscale finetune {filename}",
        "working_dir": ".",
        "image_uri": "localhost:5555/anyscale/llm-forge:0.5.6",
        "requirements": [
            "wandb",
        ],
        "env_vars": {
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
        }
    }
    compute_config_kwargs = kwargs.get("compute_config", {})
    compute_config_dict.update(compute_config_kwargs)
    compute_config = JobConfig(**compute_config_dict)

    job_runner_config_path = kwargs.get("compute_yaml_path", "job_runner_config.yaml")

    return job_runner_config_path, compute_config

def start_remote_training(compute_config, **kwargs) -> str:
    job_id: str = anyscale.job.submit(compute_config)
    return job_id

def wait_for_training(job):
    logger.info("Waiting for training to complete")
    anyscale.job.wait(id=job.job_id, timeout_s=18000)

def get_model_info(job_id):
    return anyscale.llm.model.get(job_id=job_id).to_dict()

def copy_lora_weights(storage_uri, model_info, job_id):
    try:
        from google.cloud import storage

        storage_client = storage.Client()

        bucket_name = storage_uri.split('/')[2]
        source_folder = '/'.join(storage_uri.split('/')[3:-1])
        logger.info(f"Source folder: {source_folder}")

        bucket = storage_client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=source_folder)

        subfolders = set()
        for blob in blobs:
            if '/' in blob.name[len(source_folder):]:
                subfolder = blob.name.split('/')[:-1]
                subfolders.add('/'.join(subfolder))

        base_model_id = model_info["base_model_id"]
        lora_dynamic_path = f"dspy/lora_weights/{job_id}/{base_model_id}"

        model_names = []
        for subfolder in subfolders:
            subfolder_name = subfolder.split('/')[-1]
            destination_folder = f"{lora_dynamic_path}:{subfolder_name}"
            if subfolder_name.startswith("epoch"):
                model_names.append("/".join(destination_folder.split("/")[-2:]))
            else:
                continue
            
            subfolder_blobs = bucket.list_blobs(prefix=subfolder)
            
            for blob in subfolder_blobs:
                source_blob = bucket.blob(blob.name)
                destination_blob_name = f"{destination_folder}/{blob.name.split('/')[-1]}"
                bucket.copy_blob(source_blob, bucket, destination_blob_name)
                logger.info(f"Copied {source_blob.name} to {destination_blob_name}")

        logger.info(f"All subfolders copied to: gs://{bucket_name}/{lora_dynamic_path}")
        return model_names
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e

def read_jsonl(filename):
    with open(filename, "r") as f:
        return [ujson.loads(line) for line in f]

def write_jsonl(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")

# Fine tuning utils
def openai_data_validation(dataset: List[dict[str, Any]]) -> Union[dict[str, Any], AssertionError]:
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        return format_errors
    return None

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def check_message_lengths(dataset: List[dict[str, Any]]) -> list[int]:
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    n_too_long = sum([length > 16385 for length in convo_lens])

    if n_too_long > 0:
        logger.info(f"There are {n_too_long} examples that may be over the 16,385 token limit, they will be truncated during fine-tuning.")

    if n_missing_system > 0:
        logger.info(f"There are {n_missing_system} examples that are missing a system message.")

    if n_missing_user > 0:
        logger.info(f"There are {n_missing_user} examples that are missing a user message.")

    return convo_lens

def estimate_cost(dataset: dict[str, Any], tokens_per_message=3, tokens_per_name=1, convo_lens=None):
    MAX_TOKENS_PER_EXAMPLE = 16385
    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    if convo_lens is None:
        convo_lens = check_message_lengths(dataset)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)

    logger.info(f"""The charge for finetuning is determined by the number of epochs multiplied by the number of billing tokens in the dataset. Here are the stats for this training dataset:
    num_billing_tokens: {n_billing_tokens_in_dataset}
    n_epochs: {n_epochs}
    num_total_charge_tokens: {n_epochs * n_billing_tokens_in_dataset}""")