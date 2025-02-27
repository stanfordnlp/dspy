import json
import os
from typing import Any, Dict, List, Optional

import yaml
import logging

from dspy.clients.finetune import (
    FinetuneJob,
    # TrainingMethod,
    save_data,
)
from dspy.clients.openai import openai_data_validation

try:
    # AnyScale fine-tuning requires the following additional imports
    import anyscale
    from anyscale.job import JobConfig
except ImportError:
    anyscale = None

logger = logging.getLogger(__name__)

# List of training methods supported by AnyScale
TRAINING_METHODS_ANYSCALE = [
    TrainingMethod.SFT,
]

PROVIDER_ANYSCALE = "anyscale"


def is_anyscale_model(model: str) -> bool:
    """Check if the model is an AnyScale model."""
    # TODO: This needs to be implemented to support fine-tuning
    print("Is AnyScale model is not implemented, returning False as a default to not break lm.py")
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
    assert anyscale.__version__ >= "0.24.65", "Anyscale version >= 0.24.65 is required to use the dataset upload feature"
    assert all([x in train_kwargs for x in ["job_config_path", "llmforge_config_path"]]), "Both job_config_path and llmforge_config_path are required"
    train_kwargs_copy = train_kwargs.copy()
    train_kwargs_copy["model"] = model


    job_config_path = train_kwargs.get("job_config_path", None)
    llmforge_config_path = train_kwargs.get("llmforge_config_path", None)
    serve_config_path = train_kwargs.get("serve_config_path", None)

    if train_method not in TRAINING_METHODS_ANYSCALE:
        raise NotImplementedError(f"AnyScale can only support {TRAINING_METHODS_ANYSCALE} for the time being")

    if not verify_dataset(train_data):
        # TODO: Does AnyScale support text completion models?
        err = "[Finetune] Error: Unable to verify that the dataset is in the correct format."
        logger.error(err)
        raise RuntimeError(err)

    train_data_path = save_data(train_data, provider_name=PROVIDER_ANYSCALE)

    # TODO(Isaac): Figure out a better pattern
    job_config_temp = yaml.safe_load(open(job_config_path, "r"))
    remote_train_path = submit_data(train_path=train_data_path, job_config=job_config_temp)

    job_config = generate_config_files(train_path=remote_train_path, llmforge_config_path=llmforge_config_path, job_config_path=job_config_path, model=model)

    # Remove potential duplicate compute config

    job.job_id = start_remote_training(job_config=job_config)

    wait_for_training(job.job_id)

    model_info = get_model_info(job.job_id)

    # model_info[storage_uri] is a path to your cloud where the best(last if no eval) checkpoint weights are forwarded
    storage_uri = model_info["storage_uri"]

    lora_dynamic_path = storage_uri.split(model)[0]
    final_model_name = model + storage_uri.split(model)[1]
    
    if serve_config_path:
        update_serve_model_config(lora_dynamic_path, serve_config_path)
    job.model_names = [final_model_name]

    return "openai/" + final_model_name

def wait_for_training(job_id):
    """Wait for the training to complete."""
    print("[Finetune] Waiting for training to complete...")
    anyscale.job.wait(id=job_id)
    print("[Finetune] Training completed.")


def update_serve_model_config(lora_dynamic_path: str, serve_config_path: str):
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
    print("[Finetune] Verifying dataset...")
    dataset_validation = openai_data_validation(dataset)

    if dataset_validation:
        logger.error(f"Dataset validation failed: {dataset_validation}")
        return False

    return True


def submit_data(train_path: str, job_config: Dict[str, Any]):
    """Upload the data to cloud storage."""
    print("[Finetune] Submitting data to remote storage...")
    dataset_suffix = os.path.basename(train_path).split(".")[0]
    dataset_name = f"dataset-{job_config.get('name', dataset_suffix)}"
    train_path_remote = anyscale.llm.dataset.upload(train_path, name=dataset_name, cloud=job_config.get("cloud", None)).storage_uri
    print(f"[Finetune] Data submitted. Remote train path: {train_path_remote}")

    return train_path_remote


def generate_config_files(train_path: str, llmforge_config_path: str, job_config_path: str, model: str):
    assert llmforge_config_path, "LLMForge config is required to generate the config files"
    assert job_config_path, "Job config is required to start the finetuning job"
    
    llmforge_config = yaml.safe_load(open(llmforge_config_path, "r"))
    job_config_dict = yaml.safe_load(open(job_config_path, "r"))
    
    llmforge_config["model_id"] = model
    llmforge_config["train_path"] = train_path
    llmforge_config = {k: v for k, v in llmforge_config.items() if v is not None}

    print(f"Model config data: {llmforge_config}")
    yaml.safe_dump(llmforge_config, open(llmforge_config_path, "w"))

    if not job_config_dict.get("env_vars", None):
        job_config_dict["env_vars"] = {}

    for env_var in ["HF_TOKEN", "HF_HOME", "WANDB_API_KEY"]:
        if env_var not in job_config_dict["env_vars"] and os.environ.get(env_var, None):
            job_config_dict["env_vars"][env_var] = os.environ[env_var]
    

    job_config = JobConfig(**job_config_dict)


    return job_config


def start_remote_training(job_config) -> str:
    print("[Finetune] Starting remote training...")
    job_id: str = anyscale.job.submit(job_config)
    print(f"[Finetune] Remote training started. Job ID: {job_id}")
    return job_id


def wait_for_training(job_id):
    print("Waiting for training to complete")
    anyscale.job.wait(id=job_id, timeout_s=18000)


def get_model_info(job_id):
    print("[Finetune] Retrieving model information from Anyscale Models SDK...")
    info = anyscale.llm.model.get(job_id=job_id).to_dict()
    print(f"[Finetune] Model info retrieved: {info}")
    return info

def read_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]
