from typing import Any, Optional, Literal, Union

import openai
from dsp.modules.lm import FinetuneableLM
from dsp.modules.gpt3 import GPT3
import tiktoken

from collections import defaultdict

def openai_data_validation(dataset: dict[str, Any]) -> Union[dict[str, Any], None]:
    """Validate OpenAI data before sending it to the model.

    Args:
        dataset: OpenAI data to validate

    Returns:
        Either a list of errors and their counts or None if no errors are found
    """
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
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
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
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def check_message_lengths(dataset: dict[str, Any]) -> list[int]:
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

    n_too_long = sum([l > 16385 for l in convo_lens])
    if n_too_long > 0:
        print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")

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
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class OpenAIModel(GPT3, FinetuneableLM):
    """Wrapper around specifically the OpenAI API to finetune.

        Args:
            model (str, optional): OpenAI supported LLM model to use. Defaults to "gpt-3.5-turbo-instruct".
            api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
            api_provider (Literal["openai"], optional): The API provider to use. Defaults to "openai".
            model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
            system_prompt (Optional[str], optional): The system prompt to use. Defaults to None in init, and "You are a helpful assistant." in format_data_for_vanilla_finetuning.
            fine_tuning_file_id (Optional[str], optional): The file id of the data to be used for fine-tuning. Defaults to None. Currently checkpointing is not supported but this would be the place to store it.
            fine_tuning_job_id (Optional[str], optional): The job id of the fine-tuning job. Defaults to None. Again, checkpointing is not supported but this would be the place to store it.
            **kwargs: Additional arguments to pass to the API provider.
        """
    
    def __init__(
            self,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        api_provider: Literal["openai"] = "openai",
        api_base: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        system_prompt: Optional[str] = None,
        fine_tuning_file_id: Optional[str] = None,
        fine_tuning_job_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, api_key, api_provider, api_base, model_type, system_prompt, **kwargs)
        assert self.provider == "openai", "You must use an OpenAI model with this class."
        self.fine_tuning_file_id = fine_tuning_file_id # Maybe move into abstract class if every fine tuning needs it
        self.fine_tuning_job_id = fine_tuning_job_id
    
    # https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    def verify_training_arguments(self, dataset: dict[str, Any], training_arguments: dict[str, Any]) -> bool:
        data_val = openai_data_validation(dataset)

        # training data validation
        if data_val is not None:
            print("Data validation failed")
            print(data_val)
            return False
        
        convo_lens = check_message_lengths(dataset)

        estimate_cost(dataset, convo_lens=convo_lens)
        # validation data validation TODO
        # integration

        return True

    def format_data_for_vanilla_finetuning(self, data: list[dict[str, str]]) -> list[dict[str, Any]]:
        def format_single_item(item):
            result = self._get_chat_format(item["prompt"])
            result.append({"role": "assistant", "content": item["completion"]})
            # NOTE: This system prompt is a default
            if result[0]["role"] != "system":
                result.insert(0, {"role": "system", "content": self.system_prompt or "You are a helpful assistant."})
            messages = {"messages": result}
            return messages
        return list(map(format_single_item, data))
    
    def _get_chat_format(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def _load_data(self, data_path: str) -> str:
        """Load data from a file and submit it to the OpenAI API for fine-tuning.

        Args:
            data_path (str): The path to the file containing the data.

        Returns:
            str: The file id of the data to be used for fine-tuning.
        """
        # https://platform.openai.com/docs/api-reference/files/object
        # submit file to OAI API and get a File object back
        # retain the id for future reference
        # can one lm be finetuned multiple times?
        
        file = openai.files.create(
            file=open(f"{data_path}", "rb"),
            purpose="fine-tune"
        )
        # this returns a file id that is different from the file name!
        return file.id

    # TODO: wandb integration
    # TODO: validation dataset
    def start_training(self, **kwargs) -> str: # I believe it should be the LMs responsibility to store the data
        assert self.fine_tuning_file_id is not None, "You must load data before starting training"
        
        valid_hparams = ["n_epochs", "learning_rate_multiplier", "batch_size"]
        hparams = {k: v for k, v in kwargs.items() if k in valid_hparams}

        job = openai.fine_tuning.jobs.create(
            training_file=self.fine_tuning_file_id,
            model=self.kwargs["model"],
            seed=kwargs.get("seed", 0),
            hyperparameters=hparams,
        )
        self.fine_tuning_job_id = job.id
        return job.id

    def validate_hyperparameters(self, hyperparameters: dict[str, Any]) -> bool:
        # https://platform.openai.com/docs/api-reference/fine-tuning/create#fine-tuning-create-hyperparameters
        def is_positive_number(value, convert_func):
            try:
                return convert_func(value) > 0
            except (ValueError, TypeError):
                return False

        parameters = {
            "batch_size": int,
            "n_epochs": int,
            "learning_rate": float,
        }

        for param, convert_func in parameters.items():
            value = hyperparameters.get(param, None)
            if value and not is_positive_number(value, convert_func):
                print(f"Invalid {param}: Must be a positive {convert_func.__name__}.")
                return False

        return True
    
        #  probably delete the file after training

    def load_from_job_id(self, job_id: str) -> None:
        # Load the model from the job id
        # TODOSOON: Implement this
        pass

    def stop_training(self) -> None:
        # TODOSOON: Implement this
        # # Cancel a job client.fine_tuning.jobs.cancel("ftjob-abc123")
        pass

    def check_training_status(self) -> bool:
        assert self.fine_tuning_job_id is not None, "You must start training before checking status"
        temp_job = openai.fine_tuning.jobs.retrieve(self.fine_tuning_job_id)
        if temp_job.status == "succeeded":
            return True
        elif temp_job.status == "failed":
            print("Job failed") # TODO: Handle this gracefully
            return False
        elif temp_job.status == "running":
            return False
    
    def retrieve_trained_model_client(self):
        assert self.fine_tuning_job_id is not None, "You must start training before retrieving the model"
        job = openai.fine_tuning.jobs.retrieve(self.fine_tuning_job_id)
        if job.status == "succeeded":
            self.kwargs["model"] = job.fine_tuned_model  
