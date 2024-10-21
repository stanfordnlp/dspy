from typing import Any, Dict, List, Optional
import json
import yaml
import os

from dspy.utils.logging import logger
from dspy.clients.finetune import (
    FinetuneJob,
    TrainingMethod,
    save_data,
)
from dspy.clients.openai import openai_data_validation

try:
    # AnyScale fine-tuning requires the following additional imports
    import anyscale
    from anyscale.job import JobConfig
except ImportError:
    anyscale = None


# List of training methods supported by AnyScale
TRAINING_METHODS_ANYSCALE = [
    TrainingMethod.SFT,
]

PROVIDER_ANYSCALE = "anyscale"


def is_anyscale_model(model: str) -> bool:
    """Check if the model is an AnyScale model."""
    # TODO: This needs to be implemented to support fine-tuning
    logger.info("Is AnyScale model is not implemented, returning False as a default to not break lm.py")
    return False


class FinetuneJobAnyScale(FinetuneJob):

    def __init__(self, *args, **kwargs):
        self.job_id = None
        self.model_names = None
        super().__init__(*args, **kwargs)


def finetune_anyscale(
        job: FinetuneJobAnyScale,
        model: str,
        train_data: List[Dict[str, Any]],
        train_kwargs: Optional[Dict[str, Any]]=None,
        train_method: TrainingMethod = TrainingMethod.SFT,
    ) -> str:
    """Start the finetune job."""
    train_kwargs = train_kwargs or {}
    assert "model" not in train_kwargs, "Model should not be in the train_kwargs"
    train_kwargs_copy = train_kwargs.copy()
    train_kwargs_copy["model"] = model

    if train_method not in TRAINING_METHODS_ANYSCALE:
        raise NotImplementedError(f"AnyScale can only support {TRAINING_METHODS_ANYSCALE} for the time being")

    if not verify_dataset(train_data):
        # TODO: Does AnyScale support text completion models?
        err = "[Finetune] Error: Unable to verify that the dataset is in the correct format."
        logger.error(err)
        raise RuntimeError(err)

    train_data_path = save_data(train_data, provider_name=PROVIDER_ANYSCALE)
    remote_train_path, _ = submit_data(train_path=train_data_path)

    _, compute_config = generate_config_files(train_path=remote_train_path, **train_kwargs_copy)
    # Remove potential duplicate compute config
    train_kwargs_copy.pop("compute_config")

    job.job_id = start_remote_training(compute_config=compute_config, **train_kwargs_copy)

    wait_for_training(job.job_id)

    model_info = get_model_info(job.job_id)
    # model_info[storage_uri] is a path to your cloud where the best(last if no eval) checkpoint weights are forwarded
    storage_uri = model_info["storage_uri"]

    lora_dynamic_path = storage_uri.split(model)[0]
    final_model_name = model + storage_uri.split(model)[1]

    assert "serve_config_path" in train_kwargs, "serve_config_path is required to update the model config"
    serve_config_path = train_kwargs.pop("serve_config_path")
    update_model_config(lora_dynamic_path, serve_config_path)
    job.model_names = [final_model_name]

    # NOTE: For Litellm we need to prepend "openai/" to the model name
    return "openai/" + final_model_name

def wait_for_training(job_id):
    """Wait for the training to complete."""
    logger.info("[Finetune] Waiting for training to complete...")
    anyscale.job.wait(id=job_id, timeout_s=18000)
    logger.info("[Finetune] Training completed.")


def update_model_config(lora_dynamic_path: str, serve_config_path: str):
    """Update the model config storage location with the job_id."""
    with open(serve_config_path, "r") as f:
        serve_config = yaml.safe_load(f)
    
    model_config_location = serve_config["applications"][0]["args"]["llm_configs"][0]
    
    with open(model_config_location, "r") as f:
        model_config = yaml.safe_load(f)

    model_config["lora_config"]["dynamic_lora_loading_path"] = lora_dynamic_path
    
    with open(model_config_location, "w") as f:
        yaml.safe_dump(model_config, f)
    

def verify_dataset(dataset: List[dict[str, Any]]) -> bool:
    """Verify the training arguments before starting training."""
    logger.info("[Finetune] Verifying dataset...")
    dataset_validation = openai_data_validation(dataset)

    if dataset_validation:
        logger.error(f"Dataset validation failed: {dataset_validation}")
        return False

    return True


def submit_data(train_path: str):
    """Upload the data to the Workspace cloud storage."""
    logger.info("[Finetune] Submitting data to remote storage...")
    storage = os.environ['ANYSCALE_ARTIFACT_STORAGE']
    
    datasets = {"train": train_path}

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

    logger.info(f"[Finetune] Data submitted. Remote train path: {fine_tuning_file_ids['train']}")
    
    return fine_tuning_file_ids["train"], fine_tuning_file_ids.get("val", None)


def generate_config_files(train_path: str, **kwargs):
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
        "logger": kwargs.get("logging_kwargs", {}),
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
    assert kwargs["compute_config"], "Compute config is required to start the finetuning job"
    compute_config_dict = kwargs.pop("compute_config", {
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
    })
    compute_config_dict["env_vars"] = compute_config_dict.get("env_vars", {})
    for env_var in ["WANDB_API_KEY", "HF_TOKEN", "HF_HOME"]:
        if env_var not in compute_config_dict["env_vars"]:
            compute_config_dict["env_vars"][env_var] = os.environ.get(env_var, "")

    compute_config_dict["entrypoint"] = compute_config_dict["entrypoint"].format(filename=filename)
    compute_config = JobConfig(**compute_config_dict)

    job_runner_config_path = kwargs.get("compute_yaml_path", "job_runner_config.yaml")

    return job_runner_config_path, compute_config


def start_remote_training(compute_config, **kwargs) -> str:
    logger.info("[Finetune] Starting remote training...")
    job_id: str = anyscale.job.submit(compute_config)
    logger.info(f"[Finetune] Remote training started. Job ID: {job_id}")
    return job_id


def wait_for_training(job_id):
    logger.info("Waiting for training to complete")
    anyscale.job.wait(id=job_id, timeout_s=18000)


def get_model_info(job_id):
    logger.info("[Finetune] Retrieving model information from Anyscale Models SDK...")
    info = anyscale.llm.model.get(job_id=job_id).to_dict()
    logger.info(f"[Finetune] Model info retrieved: {info}")
    return info

def read_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]
