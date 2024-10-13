from pyexpat import model
import dsp
from dsp.modules.lm import TrainableLM, TrainingMethod
from concurrent.futures import Future
from typing import Any, List, Optional, Literal, Union
import ujson
import openai
import yaml
import os
import time

try:
    from anyscale.job import JobConfig
    import anyscale
except ImportError:
    anyscale = None

class TrainableAnyscale(TrainableLM):
    """Wrapper around specifically the OpenAI API to finetune.

        Args:
            model (str, optional): OpenAI supported LLM model to use. Defaults to "gpt-3.5-turbo-instruct".
            api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
            api_provider (Literal["openai"], optional): The API provider to use. Defaults to "openai".
            model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
            system_prompt (Optional[str], optional): The system prompt to use. Defaults to None in init, and "You are a helpful assistant." in format_data_for_vanilla_finetuning.
            **kwargs: Additional arguments to pass to the API provider.
    """
    SUPPORTED_TRAINING_METHODS = [TrainingMethod.SFT] # TODO: Add DPO

    def __init__(
            self,
            model: str = "gpt-3.5-turbo-instruct", # TODO
            api_key: Optional[str] = None, # TODO
            api_provider: Literal["anyscale"] = "anyscale", # TODO
            api_base: Optional[str] = None, # TODO
            model_type: Literal["chat", "text"] = None,
            system_prompt: Optional[str] = None,
            **kwargs,
    ):
        assert api_key is None, "There is no API key needed for this provider."
        assert api_base is None, "There is no API base needed for this provider."
        assert anyscale is not None, "You must have the Anyscale SDK installed to use this class."
        api_key, api_base = "", ""
        super().__init__(model, api_key=api_key, api_provider=api_provider, api_base=api_base, model_type=model_type, system_prompt=system_prompt, **kwargs)
        assert self.provider == "anyscale", "You must use an Anyscale model with this class."
        self.fine_tuning_file_ids = {}

    def _verify_datasets(self, dataset: List[dict[str, Any]], valset: Optional[List[dict[str, Any]]]) -> bool:
        """Verify the training arguments before starting training.
        This will look for a yml template and/or list of hyperparameters and fill in kwargs with any missing values.
        The current implementation will only allow for overriding the default yaml template for the current LM model.

        Args:
            dataset: The dataset to be used for training.
            valset: The validation dataset to be used to calculate validation loss.
        """
        def validate_dataset(name, data: dict[str, Any]) -> bool:
            dataset_validation = openai_data_validation(data)

            if dataset_validation:
                print("Dataset validation failed")
                print(dataset_validation)
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
    
    def _generate_config_files(self, train_path: str, eval_path: Optional[str], **kwargs):
        # load hparams from yaml file in kwargs as dict or default for the lm
        # load hparams as dict in kwargs
        base_model_yaml_path = kwargs.get("train_config_yaml", None)
        use_lora = kwargs.get("use_lora", False)
        example_dir = ""
        lora_path= "configs/training/lora" if use_lora else "configs/training/full_param"
        if not base_model_yaml_path:
            # TODO: Add default + block ft for non-supported models
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

            default_model_yaml_path = get_yaml_config(self.kwargs["model"])

            base_model_yaml_path = os.path.join(example_dir, lora_path, default_model_yaml_path)
            print(f"Using default yaml template for model: {base_model_yaml_path}")
            
        model_config_data = yaml.safe_load(open(base_model_yaml_path, "r"))
        # Need to determine if running on head node or not
        model_config_data.update(kwargs.get("hyperparameters", {}))
        
        model_config_data["model_id"] = self.kwargs["model"]

        custom_modifications = {
            "model_id": self.kwargs["model"],
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


        # use a hash of the model_config_data as the filename
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
        # NOTE: I messed up the llama 3 8b file
        print(model_config_data)
        yaml.safe_dump(model_config_data, open(filename, "w"))

        # ft_path = kwargs.get("ft_path", None)
        # ft_path = os.path.join(example_dir, "utils", "ft.py") or ft_path
        ft_path = os.path.join("utils", "ft.py")

        # Should this be hardcoded or have a default
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
        # yaml.safe_dump(compute_config_dict, open(job_runner_config_path, "w"))

        # TODO: Validate the hyperparameters
        # if not self.validate_hyperparameters(training_arguments):
        #     return False

        return job_runner_config_path, compute_config

    def _submit_data(self, train_path: str, eval_path: Optional[str]):
        """Upload the data to the Workspace cloud storage.

        Args:
            train_path: The path to the file containing the data.
            eval_path: The path to the file containing the evaluation data.

        Returns:
            str: The file id of the data to be used for fine-tuning.
        """
        storage = os.environ['ANYSCALE_ARTIFACT_STORAGE']
        
        datasets = {"train": train_path}
        if eval_path:
            datasets["val"] = eval_path

        for name, path in datasets.items():
            num_items = len(read_jsonl(path))
            print(f"Number of items in {name} data: {num_items}")

            remote_path = os.path.join(storage, path.split("/")[-1])
            print(f"Uploading {name} data to S3 at {remote_path}")
            # NOTE: trying a local copy for now
            if remote_path[:2] == "s3":
                os.system(f"aws s3 cp {path} {remote_path}")
            elif remote_path[:2] == "gs":
                os.system(f"gcloud storage cp {path} {remote_path}")
            else:
                os.system(f"cp {path} {remote_path}")
                print(f"Copied {path} to {remote_path}")
            self.fine_tuning_file_ids[name] = remote_path
        
        return self.fine_tuning_file_ids["train"], self.fine_tuning_file_ids.get("val", None)

    def _start_remote_training(self, compute_config, **kwargs) -> str:
        job_id: str = anyscale.job.submit(
            compute_config
        )
        return job_id


    # TODO: Straight up out of scope lmao
    def validate_hyperparameters(self, hyperparameters: dict[str, Any]) -> bool:
        """Validate the hyperparameters before starting training. Only checks the hyperparameters that are allowed in the OpenAI API.
        More information on hyperparameter validation can be found here: https://platform.openai.com/docs/api-reference/fine-tuning/create#fine-tuning-create-hyperparameters

        Args:
            hyperparameters: The hyperparameters to be used for training.

            Returns:
                bool: Whether the hyperparameters are valid."""

        return True
    
    # TODO
    def stop_training(self) -> None:
        anyscale.job.cancel(self.fine_tuning_job_id)

        self.fine_tuning_job_id = None
    
    # # TODO
    # def check_training_status(self) -> bool:
    #     assert self.fine_tuning_job_id is not None, "You must start training before checking status"
    #     temp_job = openai.fine_tuning.jobs.retrieve(self.fine_tuning_job_id)
    #     if temp_job.status == "succeeded":
    #         return True
    #     elif temp_job.status == "failed":
    #         print("Job failed")
    #         raise RuntimeError(
    #             "Job failed, we recommend checking the logs and restarting the compile method")
    #     elif temp_job.status == "running":
    #         return False
    
    # # TODO
    # def retrieve_trained_model_client(self):
    #     assert self.fine_tuning_job_id is not None, "Start training before retrieving the model"
    #     job = openai.fine_tuning.jobs.retrieve(self.fine_tuning_job_id)
    #     if job.status == "succeeded":
    #         # NOTE: Not making a copy here because that is done before the training process starts
    #         self.kwargs["model"] = job.fine_tuned_model
    #     else:
    #         raise RuntimeError("Job not completed yet, cannot retrieve model")
    
    # TODO 
    # Note: working here
    def start_training(self, future: Future['TrainableOpenAI'], train_path: str, eval_path: Optional[str], method: TrainingMethod, **kwargs):
        """
        Handles the fine-tuning process for an OpenAIModel instance.

        Args:
            original_model: The original model instance to be fine-tuned.
            future: The Future object that will hold the fine-tuned model.
            **kwargs: Additional arguments for fine-tuning.
        """
        try:
            print("Starting training process...")
            if method not in self.SUPPORTED_TRAINING_METHODS:
                raise NotImplementedError(f"TrainableOpenAI can only support {TrainingMethod.SFT} for the time being")

            print("Reading and validating datasets...")
            # Convert the data from prompt completion to OAI compatible messages
            # This can be slightly flexible to accept json or jsonl and to check if the key starts with "user/assistant/system" or if first key is "messages"
            train_dataset = read_jsonl(train_path)
            val_dataset = read_jsonl(eval_path) if eval_path else None

            # This is where we validate the yaml
            if not self._verify_datasets(train_dataset, val_dataset):
                print("Error: Unable to verify arguments")
                raise RuntimeError("Unable to verify argument")
            
            # TODO: Validate kwargs

            if method != TrainingMethod.SFT:
                raise NotImplementedError("Only SFT finetuning is supported at the moment.")

            print("Converting data to JSONL format...")
            # Convert data into jsonl format
            for path, dataset in [(train_path, train_dataset), (eval_path, val_dataset)]:
                if not (path and dataset):
                    continue
                with open(path, "w") as f:
                    for item in dataset:
                        f.write(ujson.dumps(item) + "\n")

            print("Submitting data to remote storage...")
            remote_train_path, remote_eval_path = self._submit_data(train_path=train_path, eval_path=eval_path)
            print(f"Data submitted. Remote train path: {remote_train_path}, Remote eval path: {remote_eval_path}")

            print("Generating configuration files...")
            _, compute_config = self._generate_config_files(train_path=remote_train_path, eval_path=remote_eval_path, **kwargs)
            
            print("Starting remote training...")
            # Start the remote training
            job_id = self._start_remote_training(compute_config=compute_config, **kwargs)
            self.fine_tuning_job_id = job_id
            print(f"Remote training started. Job ID: {job_id}")

            print("Waiting for training to complete...")
            # Wait for the training to complete
            self.wait_for_training()
            print("Training completed.")

            print("Retrieving model information...")
            model_info = anyscale.llm.model.get(job_id=job_id).to_dict()
            print(f"Model info retrieved: {model_info}")

            storage_uri = model_info["storage_uri"]
            print(f"Copying LoRA weights from {storage_uri}...")
            model_names = copy_lora_weights(storage_uri, model_info, job_id)
            print(f"LoRA weights copied. Model names: {model_names}")

            print("Setting result in future object...")
            future.set_result(model_names)
            print("Training process completed successfully.")

        except Exception as e:
            print(f"Error occurred during training: {str(e)}")
            future.set_exception(e)

    def wait_for_training(self):
        print("Waiting for training to complete")
        anyscale.job.wait(id=self.fine_tuning_job_id, timeout_s=18000) 
    
    def get_finetune(self, method: TrainingMethod, train_path: str, eval_path: Optional[str], **kwargs) -> Future[TrainableLM]:
        """
        Does everything required to finetune an anyscale model.

        This includes:
        - Convert the data to the required format
        - Validate the data
        - Submit the data to the cloud storage
        - Start the remote training
        - Wait for the training to complete
        - Retrieve the trained model

        Args:
            train_path: The path to the training data.
            val_path: The path to the validation data.
            method: The training method to use.
            **kwargs: Additional arguments to pass to the API provider.
                # TODO add kwargs
        Returns:
            Future[TrainableLM]: A Future object that will hold the fine-tuned model
        """
        return super().get_finetune(train_path=train_path, eval_path=eval_path, method=method, **kwargs)


# TODO: This should be moved into the TrainableAnyscaleLM class

def copy_lora_weights(storage_uri, model_info, job_id):
    """
    Copies LoRA weights from a source GCS bucket to a destination folder.
    
    Args:
    storage_uri (str): The GCS URI of the source bucket and folder.
    model_info (dict): A dictionary containing model information, including 'base_model_id'.
    job_id (str): The ID of the job associated with the LoRA weights.
    
    Returns:
    list: A list of copied model names.
    """
    try:
        from google.cloud import storage

        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()

        # Parse the storage_uri to get bucket name and source folder
        bucket_name = storage_uri.split('/')[2]
        source_folder = '/'.join(storage_uri.split('/')[3:-1])
        print(f"Source folder: {source_folder}")

        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # List all subfolders in the source folder
        blobs = bucket.list_blobs(prefix=source_folder)

        subfolders = set()
        for blob in blobs:
            if '/' in blob.name[len(source_folder):]:
                subfolder = blob.name.split('/')[:-1]
                subfolders.add('/'.join(subfolder))

        # Construct the destination folder path
        base_model_id = model_info["base_model_id"]
        lora_dynamic_path = f"dspy/lora_weights/{job_id}/{base_model_id}"

        model_names = []
        # Copy each subfolder to the main storage folder
        for subfolder in subfolders:
            subfolder_name = subfolder.split('/')[-1]
            destination_folder = f"{lora_dynamic_path}:{subfolder_name}"
            if subfolder_name.startswith("epoch"):
                model_names.append("/".join(destination_folder.split("/")[-2:]))
            else:
                continue
            
            # List all blobs in the subfolder
            subfolder_blobs = bucket.list_blobs(prefix=subfolder)
            
            # Copy each blob to the destination folder
            for blob in subfolder_blobs:
                source_blob = bucket.blob(blob.name)
                destination_blob_name = f"{destination_folder}/{blob.name.split('/')[-1]}"
                bucket.copy_blob(source_blob, bucket, destination_blob_name)
                print(f"Copied {source_blob.name} to {destination_blob_name}")

        print(f"All subfolders copied to: gs://{bucket_name}/{lora_dynamic_path}")
        return model_names
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

def read_jsonl(filename):
    with open(filename, "r") as f:
        return [ujson.loads(line) for line in f]

def write_jsonl(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")

# Fine tuning utils - TODO: Find a better spot for these

#-------------------------------------------------------------------------------
#    Templates for the user-facing strings used by this module
#-------------------------------------------------------------------------------

_ERR_MSG_DATASET_VALIDATION = """Found errors in the dataset format using the \
OpenAI API. Here are the number of datapoints for each error type found:
{err_info}"""

_ERR_MSG_DATASET_VALIDATION_TYPE = """    {key}: {val}"""

_INFO_MSG_DATASET_VALIDATION = """No errors found in the dataset format \
using the OpenAI API."""

_INFO_MSG_DATAPOINT_LONG = """There are {num} examples that may be over the \
16,385 token limit, they will be truncated during fine-tuning."""

_INFO_MSG_DATAPOINT_SYSTEM = """There are {num} examples that are missing a \
system message."""

_INFO_MSG_DATAPOINT_USER = """There are {num} examples that are missing a \
user message."""

_INFO_MSG_TRAINING = """The charge for finetuning is determined by the number \
of epochs multiplied by the number of billing tokens in the dataset. Here are \
the stats for this training dataset:
    num_billing_tokens: {num_billing_tokens}
    n_epochs: {n_epochs}
    num_total_charge_tokens: {training_charge}"""

_INFO_DATA_FILE_UPLOAD = "Uploaded the data file {fname} to the OpenAI servers."

_INFO_MSG_TRAINING_STARTED = "Started training with the following ID: {job_id}"


#-------------------------------------------------------------------------------
#    Helper functions
#-------------------------------------------------------------------------------

# These utility functions come from: https://cookbook.openai.com/examples/chat_finetuning_data_prep
def openai_data_validation(dataset: List[dict[str, Any]]) -> Union[dict[str, Any], AssertionError]:
    """Validate OpenAI data before sending it to the model.

    Args:
        dataset: OpenAI data to validate

    Returns:
        Either a list of errors and their counts or None if no errors are found
    """
    # TODO: Move the import outside the function

    # TODO: Counting the number of errors is not very useful, we can consider
    # raising an error as we run into issues.
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

    # Raise an error if there are any format errors
    if format_errors:
        err_info = ""
        for k, v in format_errors.items():
            err_info += _ERR_MSG_DATASET_VALIDATION_TYPE.format(key=k, val=v)

        err_msg = _ERR_MSG_DATASET_VALIDATION.format(err_info=err_info)
        raise ValueError(err_msg)
        
    # If no errors are found, log a message
    msg = _INFO_MSG_DATASET_VALIDATION
    print(msg)


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    # TODO: Should the import be moved outside? Same with the other functions
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
    # TODO: Move the import outside the function

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
        assistant_message_lens.append(
            num_assistant_tokens_from_messages(messages))
    n_too_long = sum([length > 16385 for length in convo_lens])

    if n_too_long > 0:
        msg = _INFO_MSG_DATAPOINT_LONG.format(num=n_too_long)
        print(msg)

    if n_missing_system > 0:
        msg = _INFO_MSG_DATAPOINT_SYSTEM.format(num=n_missing_system)
        print(msg)

    if n_missing_user > 0:
        msg = _INFO_MSG_DATAPOINT_USER.format(num=n_missing_user)
        print(msg)

    return convo_lens


def estimate_cost(dataset: dict[str, Any], tokens_per_message=3, tokens_per_name=1, convo_lens=None):
    # TODO: Move the import outside the function

    MAX_TOKENS_PER_EXAMPLE = 16385

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    # TODO: Can we not fix the above variables as constants?
    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS,
                       MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS,
                       MAX_TARGET_EXAMPLES // n_train_examples)

    if convo_lens is None:
        convo_lens = check_message_lengths(dataset)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)

    # TODO would be more informative to share the total price
    msg = _INFO_MSG_TRAINING.format(
        num_billing_tokens=n_billing_tokens_in_dataset,
        n_epochs=n_epochs,
        training_charge=n_epochs * n_billing_tokens_in_dataset
    )
    print(msg)