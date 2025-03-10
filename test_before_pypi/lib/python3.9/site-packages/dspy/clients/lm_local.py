import datetime
import logging
import random
import socket
import string
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, save_data

if TYPE_CHECKING:
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)


class LocalProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJob

    @staticmethod
    def launch(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        try:
            import sglang  # noqa: F401
        except ImportError:
            raise ImportError(
                "For local model launching, please install sglang by running "
                '`pip install "sglang[all]"; pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/`'
            )

        if hasattr(lm, "process"):
            logger.info("Server is already launched.")
            return

        launch_kwargs = launch_kwargs or lm.launch_kwargs

        import os

        model = lm.model
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]
        if model.startswith("huggingface/"):
            model = model[len("huggingface/"):]

        logger.info(f"Grabbing a free port to launch an SGLang server for model {model}")
        logger.info(
            f"We see that CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        port = get_free_port()
        timeout = launch_kwargs.get("timeout", 1800)
        command = f"python -m sglang.launch_server --model-path {model} --port {port} --host 0.0.0.0"

        # We will manually stream & capture logs.
        process = subprocess.Popen(
            command.replace("\\\n", " ").replace("\\", " ").split(),
            text=True,
            stdout=subprocess.PIPE,  # We'll read from pipe
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
        )

        # A threading.Event to control printing after the server is ready.
        # This will store *all* lines (both before and after readiness).
        logger.info(f"SGLang server process started with PID {process.pid}.")
        stop_printing_event = threading.Event()
        logs_buffer = []

        def _tail_process(proc, buffer, stop_event):
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    # Process ended and no new line
                    break
                if line:
                    buffer.append(line)
                    # Print only if stop_event is not set
                    if not stop_event.is_set():
                        print(line, end="")

        # Start a background thread to read from the process continuously
        thread = threading.Thread(
            target=_tail_process,
            args=(process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()

        # Wait until the server is ready (or times out)
        base_url = f"http://localhost:{port}"
        try:
            wait_for_server(base_url, timeout=timeout)
        except TimeoutError:
            # If the server doesn't come up, we might want to kill it:
            process.kill()
            raise

        # Once server is ready, we tell the thread to stop printing further lines.
        stop_printing_event.set()

        # A convenience getter so the caller can see all logs so far (and future).
        def get_logs() -> str:
            # Join them all into a single string, or you might return a list
            return "".join(logs_buffer)

        # Let the user know server is up
        logger.info(
            f"Server ready on random port {port}! Logs are available via lm.get_logs() method on returned lm."
        )

        lm.kwargs["api_base"] = f"http://localhost:{port}/v1"
        lm.kwargs["api_key"] = "local"
        lm.get_logs = get_logs
        lm.process = process
        lm.thread = thread


    @staticmethod
    def kill(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        from sglang.utils import terminate_process
        if not hasattr(lm, "process"):
            logger.info("No running server to kill.")
            return
        # Ideally, the following happens atomically
        terminate_process(lm.process)
        lm.thread.join()
        del lm.process
        del lm.thread
        del lm.get_logs
        logger.info("Server killed.")

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[TrainDataFormat],
        train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]

        if train_data_format != TrainDataFormat.CHAT:
            raise ValueError("Only chat models are supported for local finetuning.")

        data_path = save_data(train_data)
        output_dir = create_output_dir(model, data_path)

        default_train_kwargs = {
            "device": None,
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
        }
        train_kwargs={**default_train_kwargs, **(train_kwargs or {})}
        output_dir = train_kwargs["output_dir"]  # user might have changed the output_dir

        logger.info(f"Starting local training, will save to {output_dir}")
        train_sft_locally(
            model_name=model,
            train_data=train_data,
            train_kwargs=train_kwargs,
        )

        logger.info("Training complete")
        return f"openai/local:{output_dir}"


def create_output_dir(model_name, data_path):
    model_str = model_name.replace("/", "-")
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rnd_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    model_identifier = f"{rnd_str}_{model_str}_{time_str}"
    output_dir = data_path.replace(".jsonl", "_" + model_identifier)
    return output_dir


def train_sft_locally(model_name, train_data, train_kwargs):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer, setup_chat_format
    except ImportError:
        raise ImportError(
            "For local finetuning, please install torch, transformers, and trl "
            "by running `pip install -U torch transformers accelerate trl peft`"
        )

    device = train_kwargs.get("device", None)
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    logger.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Set up the chat format; generally only for non-chat model variants, hence the try-except.
    try:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    except Exception:
        pass

    if tokenizer.pad_token_id is None:
        logger.info("Adding pad token to tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

    logger.info("Creating dataset")
    if "max_seq_length" not in train_kwargs:
        train_kwargs["max_seq_length"] = 4096
        logger.info(f"The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to {train_kwargs['max_seq_length']}")

    from datasets import Dataset

    hf_dataset = Dataset.from_list(train_data)
    def tokenize_function(example):
        return encode_sft_example(example, tokenizer, train_kwargs["max_seq_length"])
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type="torch")
    tokenized_dataset = tokenized_dataset.filter(lambda example: (example["labels"] != -100).any())

    USE_PEFT = train_kwargs.get("use_peft", False)
    peft_config = None

    if USE_PEFT:
        from peft import LoraConfig

        rank_dimension = 32
        lora_alpha = 64
        lora_dropout = 0.05

        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    sft_config = SFTConfig(
        output_dir=train_kwargs["output_dir"],
        num_train_epochs=train_kwargs["num_train_epochs"],
        per_device_train_batch_size=train_kwargs["per_device_train_batch_size"],
        gradient_accumulation_steps=train_kwargs["gradient_accumulation_steps"],
        learning_rate=train_kwargs["learning_rate"],
        max_grad_norm=2.0,  # note that the current SFTConfig default is 1.0
        logging_steps=20,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_steps=10_000,
        bf16=train_kwargs["bf16"],
        max_seq_length=train_kwargs["max_seq_length"],
        packing=train_kwargs["packing"],
        dataset_kwargs={  # We need to pass dataset_kwargs because we are processing the dataset ourselves
            "add_special_tokens": False,  # Special tokens handled by template
            "append_concat_token": False,  # No additional separator needed
        },
    )

    logger.info("Starting training")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
    )

    # Train!
    trainer.train()

    # Save the model!
    trainer.save_model()

    MERGE = True
    if USE_PEFT and MERGE:
        from peft import AutoPeftModelForCausalLM

        # Load PEFT model on CPU
        model_ = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=sft_config.output_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        merged_model = model_.merge_and_unload()
        merged_model.save_pretrained(
            sft_config.output_dir, safe_serialization=True, max_shard_size="5GB"
        )

    # Clean up!
    import gc

    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return sft_config.output_dir


def get_free_port() -> int:
    """
    Return a free TCP port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """
    Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server (e.g. http://localhost:1234)
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                # A small extra sleep to ensure server is fully up.
                time.sleep(5)
                break

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            # Server not up yet, wait and retry
            time.sleep(1)


def encode_sft_example(example, tokenizer, max_seq_length):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.

    Code obtained from the allenai/open-instruct repository: https://github.com/allenai/open-instruct/blob/4365dea3d1a6111e8b2712af06b22a4512a0df88/open_instruct/finetune.py
    """
    import torch

    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }
