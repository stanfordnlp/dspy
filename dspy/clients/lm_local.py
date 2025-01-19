import time
import socket
import requests
import threading
import subprocess

from datasets import Dataset
from typing import Any, Dict, List, Optional
from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import DataFormat, TrainingStatus, save_data


class LocalTrainingJob(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = None

    def cancel(self):
        raise NotImplementedError

    def status(self) -> TrainingStatus:
        raise NotImplementedError


class LocalProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = LocalTrainingJob

    @staticmethod
    def launch(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        try:
            import sglang
        except ImportError:
            raise ImportError(
                "For local model launching, please install sglang by running "
                '`pip install "sglang[all]"; pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/`'
            )

        if launch_kwargs is None:
            launch_kwargs = {}

        import os

        model = lm.model
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]

        print(f"Grabbing a free port to launch an SGLang server for model {model}")
        print(
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
        print(f"SGLang server process started with PID {process.pid}.")
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
        print(
            f"Server ready on random port {port}! Logs are available via lm.get_logs() method on returned lm."
        )

        lm.kwargs["api_base"] = f"http://localhost:{port}/v1"
        lm.kwargs["api_key"] = "local"
        lm.process = process
        lm.get_logs = get_logs

    @staticmethod
    def kill(model, kill_kwargs: Optional[Dict[str, Any]] = None):
        from sglang.utils import terminate_process

        terminate_process(model.process)

    @staticmethod
    def finetune(
        job: LocalTrainingJob,
        model: str,
        train_data: List[Dict[str, Any]],
        train_kwargs: Optional[Dict[str, Any]] = None,
        data_format: Optional[DataFormat] = None,
    ) -> str:
        data_path = save_data(train_data)

        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]

        model_path = data_path.replace(".jsonl", f"__{model.replace('/', '__')}")
        print(f"[Local Provider] Data saved to {data_path}")

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
            "output_dir": model_path,
        }

        print(f"[Local Provider] Starting local training, will save to {model_path}")
        train_sft_locally(
            model_name=model,
            train_data=train_data,
            train_kwargs={**default_train_kwargs, **(train_kwargs or {})},
        )

        print("[Local Provider] Training complete")
        return f"openai/local:{model_path}"


def train_sft_locally(model_name, train_data, train_kwargs):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer, setup_chat_format
        from trl import apply_chat_template
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
    print(f"Using device: {device}")

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
        print("Adding pad token to tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

    print("Creating dataset")
    trainset_dict = {
        "prompt": [entry["messages"][:-1] for entry in train_data],
        "completion": [[entry["messages"][-1]] for entry in train_data],
    }

    trainset = Dataset.from_dict(trainset_dict)
    trainset = trainset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

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
        output_dir=output_dir,
        num_train_epochs=train_kwargs["num_train_epochs"],
        per_device_train_batch_size=train_kwargs["per_device_train_batch_size"],
        gradient_accumulation_steps=train_kwargs["gradient_accumulation_steps"],
        learning_rate=train_kwargs["learning_rate"],
        max_grad_norm=2.0,
        logging_steps=20,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_steps=10_000,
        bf16=train_kwargs["bf16"],
        max_seq_length=train_kwargs["max_seq_length"],
        packing=train_kwargs["packing"],
        dataset_kwargs={
            "add_special_tokens": False,  # Special tokens handled by template
            "append_concat_token": False,  # No additional separator needed
        },
    )

    print("Starting training")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=trainset,
        peft_config=peft_config,
        processing_class=tokenizer,
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

    return output_dir


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
