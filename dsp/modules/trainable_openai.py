from collections import defaultdict
from concurrent.futures import Future
import time
from typing import Any, Dict, List, Literal, Optional, Union
import ujson

import openai

from dsp.modules.lm import TrainableLM, TrainingMethod
from dsp.modules.gpt3 import GPT3


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


#-------------------------------------------------------------------------------
#    Classes
#-------------------------------------------------------------------------------

class TrainableOpenAI(GPT3, TrainableLM):
    """Wrapper around specifically the OpenAI API to finetune."""
    SUPPORTED_TRAINING_METHODS = [TrainingMethod.SFT]

    def __init__(
            self,
            model: str = "gpt-3.5-turbo-instruct",
            api_key: Optional[str] = None,
            api_provider: Literal["openai"] = "openai",
            api_base: Optional[str] = None,
            model_type: Literal["chat", "text"] = None,
            system_prompt: Optional[str] = None,
            **kwargs,
    ):
        """Initialize the TrainableOpenAI class.
        
        Args:
            model (str, optional): OpenAI supported LLM model to use. Defaults to "gpt-3.5-turbo-instruct".
            api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
            api_provider (Literal["openai"], optional): The API provider to use. Defaults to "openai".
            model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
            system_prompt (Optional[str], optional): The system prompt to use. Defaults to None in init, and "You are a helpful assistant." in format_data_for_vanilla_finetuning.
            **kwargs: Additional arguments to pass to the API provider.
        """
        super().__init__(model, api_key=api_key, api_provider=api_provider, api_base=api_base, model_type=model_type, system_prompt=system_prompt, **kwargs)
        assert self.provider == "openai", "You must use an OpenAI model with this class."
        self.active_finetuning_job_id: str = None
        self.active_finetuning_file_ids: Optional[Dict[str, str]] = None

    def _verify_training_arguments(self, dataset: List[dict[str, Any]], **kwargs) -> bool:
        """Verify the training arguments before starting training.

        More information on dataset verification can be found here:
        https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset

        Args:
            dataset: The dataset to be used for training.
            **kwargs: The hyperparameters to be used for training.
            """
        def validate_dataset(name, data: dict[str, Any]) -> bool:
            dataset_validation = openai_data_validation(data)

            if name == "train":
                convo_lens = check_message_lengths(data)
                estimate_cost(data, convo_lens=convo_lens)

            return True

        datasets = {"train": dataset}

        for name, data in datasets.items():
            if not validate_dataset(name, data):
                return False

        valid_hparams = ["n_epochs", "learning_rate_multiplier", "batch_size"]
        training_arguments = kwargs.get("hyperparameters", {})
        training_arguments = {
            k: v for k, v in training_arguments.items() if k in valid_hparams}

        # NOTE: Not validating seed or suffix or integration

        if not self.validate_hyperparameters(training_arguments):
            return False

        return True

    def _format_data_for_vanilla_finetuning(self, data_path: str) -> List[dict[str, Any]]:
        """Convert the data from prompt completion to OAI compatible messages."""
        with open(data_path, "r", encoding="utf-8") as file:
            data = ujson.load(file)
        
        def format_single_item(item):
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]}
            ]
            # Always prepend the system prompt if available
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            
            return {"messages": messages}
        
        return list(map(format_single_item, data))

    def _submit_data(self, train_path: str, eval_path: Optional[str] = None):
        """Submit the data files to OpenAI API to be later used for fine-tuning.

        API reference: https://platform.openai.com/docs/api-reference/files/object

        Args:
            train_data: The path to the file containing the data.

        Returns:
            str: The file id of the data to be used for fine-tuning.
        """
        # TODO: Move the import outside the function

        datasets = {"train": train_path}
        if eval_path:
            datasets["val"] = eval_path

        self.active_finetuning_file_ids = {}
        for name, path in datasets.items():
            file = openai.files.create(
                file=open(f"{path}", "rb"),
                purpose="fine-tune"
            )
            msg = _INFO_DATA_FILE_UPLOAD.format(fname=path)
            print(msg)
            self.active_finetuning_file_ids[name] = file.id

    # TODO: support OAI wandb integration
    def _start_remote_training(self, **kwargs) -> str:
        # TODO: Move the import outside the function

        assert self.active_finetuning_file_ids and self.active_finetuning_file_ids["train"] is not None, "You must load data before starting training"
        hyperparameters = kwargs.get("hyperparameters", {})
        job = openai.fine_tuning.jobs.create(
            training_file=self.active_finetuning_file_ids["train"],
            model=self.kwargs["model"],
            seed=kwargs.get("seed", None),
            hyperparameters=hyperparameters,
            validation_file=self.active_finetuning_file_ids.get("val", None),
            integrations=kwargs.get("integrations", None),
            suffix=kwargs.get("suffix", None),
        )
        self.active_finetuning_job_id = job.id

        # TODO: Does this actually start the training or just create the job
        # to be put in a queue?
        msg = _INFO_MSG_TRAINING_STARTED.format(job_id=self.active_finetuning_job_id)
        print(msg)

        return job.id
    
    def _delete_active_data_files(self):
        if self.active_finetuning_file_ids is not None:
            # TODO: Handle the case where the file is already deleted
            for file in self.active_finetuning_file_ids.values():
                openai.files.delete(file)
            self.active_finetuning_file_ids = None
    
    def _delete_active_job(self):
        if self.active_finetuning_job_id is not None:
            # TODO: Handle the case where the job is already completed
            openai.fine_tuning.jobs.cancel(self.active_finetuning_job_id)
            self.active_finetuning_job_id = None

    def validate_hyperparameters(self, hyperparameters: dict[str, Any]) -> bool:
        """Validate the hyperparameters before starting training. Only checks the hyperparameters that are allowed in the OpenAI API.
        More information on hyperparameter validation can be found here: https://platform.openai.com/docs/api-reference/fine-tuning/create#fine-tuning-create-hyperparameters

        Args:
            hyperparameters: The hyperparameters to be used for training.

            Returns:
                bool: Whether the hyperparameters are valid."""
        def is_positive_number(value, convert_func):
            try:
                return convert_func(value) > 0
            except (ValueError, TypeError):
                return False

        parameters = {
            "batch_size": int,
            "n_epochs": int,
            "learning_rate_multiplier": float,
        }

        for param, convert_func in parameters.items():
            value = hyperparameters.get(param, None)
            if value and not is_positive_number(value, convert_func):
                print(
                    f"Invalid {param}: Must be a positive {convert_func.__name__}.")
                return False

        return True

    def check_training_status(self) -> bool:
        assert self.active_finetuning_job_id is not None, "You must start training before checking status"
        temp_job = openai.fine_tuning.jobs.retrieve(self.active_finetuning_job_id)
        if temp_job.status == "succeeded":
            return True
        elif temp_job.status == "failed":
            print("Job failed")
            raise RuntimeError(
                "Job failed, we recommend checking the logs and restarting the compile method")
        elif temp_job.status == "running":
            return False

    def retrieve_trained_model_client(self):
        assert self.active_finetuning_job_id is not None, "Start training before retrieving the model"
        job = openai.fine_tuning.jobs.retrieve(self.active_finetuning_job_id)

        if job.status == "succeeded":
            # NOTE: Not making a copy here because that is done before the training process starts
            self.kwargs["model"] = job.fine_tuned_model
        else:
            raise RuntimeError("Job not completed yet, cannot retrieve model")
    
        # Clean up the data files and jobs; this allows for proper caching
        self._delete_active_data_files()
        self.active_finetuning_job_id = None

    def wait_for_training(self):
        while not self.check_training_status():
            time.sleep(60)

    @staticmethod
    def load_from_job_id(job_id: str, **kwargs):
        """Load a model from a job id.

        Args:
            job_id: The job id of the fine-tuning job.
            **kwargs: Additional arguments to pass to the API provider.
        """
        job = openai.fine_tuning.jobs.retrieve(job_id)
        if job.status != "succeeded":
            raise RuntimeError("Job not completed yet, cannot retrieve model")
        model = TrainableOpenAI(model=job.fine_tuned_model, **kwargs)

        return model

    def get_finetune(self, method: TrainingMethod, train_path: str, **kwargs) -> Future[TrainableLM]:
        """Get a future for a finetuned model from the given training data.

        Args:
            method: The training method to use.
            train_path: The path to the training data, which should be in the
                format required by the training method. The format is verified
                in FinetunableLM.get_finetune(...) using the
                dsp.modules.lm.verify_training_method_data_format(...) function.
            **kwargs: Additional arguments that will be passed to the OpenAI's
                finetuning creating endpoint. Refer to the documentation at
                https://platform.openai.com/docs/api-reference/fine-tuning/create
                for details.
                
                Some possible kwargs are shared below, but refer to the
                documentation for a complete list:
                - hyperparameters (Dict[str, Any]): A dictionary containing the
                  hyperparameters to be used for training. Some example keys
                  are:
                    - n_epochs (int)
                    - learning_rate_multiplier (float)
                    - batch_size (int)
                - seed (int): The seed to use for training.
        """
        return super().get_finetune(method, train_path, **kwargs)

    def start_training(self, future: Future['TrainableOpenAI'], method: TrainingMethod, train_path: str, **kwargs):
        """Start the training process for an TrainableOpenAI model."""
        try:
            if method not in self.SUPPORTED_TRAINING_METHODS:
                raise NotImplementedError(f"TrainableOpenAI can only support {TrainingMethod.SFT} for the time being")

            # Convert the data from prompt completion to OAI compatible messages
            train_dataset = self._format_data_for_vanilla_finetuning(train_path)
            
            if not self._verify_training_arguments(train_dataset, **kwargs):
                print("Unable to verify arguments")
                raise RuntimeError("Unable to verify argument")
            
            if method != TrainingMethod.SFT:
                raise NotImplementedError("Only SFT finetuning is supported at the moment.")

            # TODO: This function should not overwrite the same file
            # TODO: Where should the intermediary files be stored? In DSPy
            # cachedir?
            with open(train_path, "w") as f:
                for item in train_dataset:
                    f.write(ujson.dumps(item) + "\n")

            self._submit_data(train_path)

            # Start the remote training
            job_id = self._start_remote_training(**kwargs)

            # Wait for the training to complete
            self.wait_for_training()

            # Retrieve the trained model and return a copy
            self.retrieve_trained_model_client()
    
            future.set_result(self)

        except Exception as e:
            future.set_exception(e)

    def stop_training(self) -> None:
        """Stop the any training process related to this instance."""
        # TODO: This is impossible to call right now because the training is
        # is wrapped in a future object.
        self._delete_active_data_files()
        self._delete_active_job()