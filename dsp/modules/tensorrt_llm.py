from pathlib import Path
from typing import Any, Optional, Union

from dsp.modules.lm import LM

## Utility functions to load models


def load_tensorrt_model(
    engine_dir: Union[str, Path],
    use_py_session: Optional[bool] = False,
    **kwargs,
) -> tuple[Any, dict]:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp

    runtime_rank = tensorrt_llm.mpi_rank()
    runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
    runner_kwargs = {
        "engine_dir": engine_dir,
        "lora_dir": kwargs.get("lora_dir", None),
        "rank": runtime_rank,
        "lora_ckpt_source": kwargs.get("lora_ckpt_source", "hf"),
    }

    if not use_py_session:
        engine_cpp_kwargs = {}
        defaults = {
            "max_batch_size": 1,
            "max_input_len": 1024,
            "max_output_len": 1024,
            "max_beam_width": 1,
            "max_attention_window_size": None,
            "sink_token_length": None,
        }

        for key, value in defaults.items():
            engine_cpp_kwargs[key] = kwargs.get(key, value)
        runner_kwargs.update(**engine_cpp_kwargs)

    runner = runner_cls.from_dir(**runner_kwargs)
    return runner, runner_kwargs


def tokenize(prompt: Union[list[dict], str], tokenizer: Any, **kwargs) -> list[int]:
    defaults = {
        "add_special_tokens": False,
        "max_input_length": 1024,
        "model_name": None,
        "model_version": None,
    }
    if not isinstance(prompt, str):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    input_ids = [
        tokenizer.encode(
            prompt,
            add_special_tokens=kwargs.get("add_special_tokens", defaults["add_special_tokens"]),
            truncation=True,
            max_length=kwargs.get("max_input_length", defaults["max_input_length"]),
        ),
    ]
    if (
        kwargs.get("model_name", defaults["model_name"]) == "ChatGLMForCausalLM"
        and kwargs.get("model_version", defaults["model_version"]) == "glm"
    ):
        input_ids.append(tokenizer.stop_token_id)
    return input_ids


class TensorRTModel(LM):
    """TensorRT integration for dspy LM."""

    def __init__(self, model_name_or_path: str, engine_dir: str, **engine_kwargs: dict) -> None:
        """Initialize the TensorRTModel.

        Args:
            model_name_or_path (str): The Huggingface ID or the path where tokenizer files exist.
            engine_dir (str): The folder where the TensorRT .engine file exists.
            **engine_kwargs (Optional[dict]): Additional engine loading keyword arguments.

        Keyword Args:
            use_py_session (bool, optional): Whether to use a Python session or not. Defaults to False.
            lora_dir (str): The directory of LoRA adapter weights.
            lora_task_uids (list[str]): list of LoRA task UIDs; use -1 to disable the LoRA module.
            lora_ckpt_source (str): The source of the LoRA checkpoint.

            If use_py_session is set to False, the following kwargs are supported:
                max_batch_size (int, optional): The maximum batch size. Defaults to 1.
                max_input_len (int, optional): The maximum input context length. Defaults to 1024.
                max_output_len (int, optional): The maximum output context length. Defaults to 1024.
                max_beam_width (int, optional): The maximum beam width, similar to `n` in OpenAI API. Defaults to 1.
                max_attention_window_size (int, optional): The attention window size that controls the
                    sliding window attention / cyclic KV cache behavior. Defaults to None.
                sink_token_length (int, optional): The sink token length. Defaults to 1.
        """
        # Implementation here
        self.model_name_or_path, self.engine_dir = model_name_or_path, engine_dir
        super().__init__(model=self.model_name_or_path)
        try:
            import tensorrt_llm
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install tensorrt-llm to use TensorRTModel",
            ) from exc

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install torch and transformers ",
                "pip install transformers==4.38.2",
            ) from exc

        # Configure tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            use_fast=True,
        )

        self.pad_id = (
            self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        )
        self.end_id = self.tokenizer.eos_token_id

        # Configure TensorRT
        self.runtime_rank = tensorrt_llm.mpi_rank()
        self.runner, self._runner_kwargs = load_tensorrt_model(engine_dir=self.engine_dir, **engine_kwargs)
        self.history: list[dict[str, Any]] = []

    def _generate(self, prompt: Union[list[dict[str, str]], str], **kwargs: dict) -> tuple[list[str], dict]:
        import torch

        input_ids = tokenize(prompt=prompt, tokenizer=self.tokenizer, **kwargs)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        run_kwargs = {}
        defaults = {
            "max_new_tokens": 1024,
            "max_attention_window_size": None,
            "sink_token_length": None,
            "end_id": self.end_id,
            "pad_id": self.pad_id,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "num_beams": 1,
            "length_penalty": 1.0,
            "early_stopping": 1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop_words_list": None,
            "bad_words_list": None,
            "streaming": False,
            "return_dict": True,
            "output_log_probs": False,
            "output_cum_log_probs": False,
            "output_sequence_lengths": True,
        }

        for k, v in defaults.items():
            run_kwargs[k] = kwargs.get(k, v)

        with torch.no_grad():
            outputs = self.runner.generate(input_ids, **run_kwargs)
        input_lengths = [x.size(0) for x in input_ids]

        output_ids, sequence_lengths = outputs["output_ids"], outputs["sequence_lengths"]

        # In case of current version of dspy it will always stay as 1
        _, num_beams, _ = output_ids.size()
        batch_idx, beams = 0, []

        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
            output_text = self.tokenizer.decode(outputs)
            beams.append(output_text)

        return beams, run_kwargs

    def basic_request(self, prompt, **kwargs: dict) -> list[str]:
        raw_kwargs = kwargs
        response, all_kwargs = self._generate(prompt, **kwargs)
        history = {
            "prompt": prompt,
            "response": response,
            "raw_kwargs": raw_kwargs,
            "kwargs": all_kwargs,
        }
        self.history.append(history)
        return response

    def __call__(
        self,
        prompt: Union[list[dict[str, str]], str],
        **kwargs,
    ):
        """TensorRTLLM generate method in dspy.

        Args:
            prompt (Union[list[dict[str, str]], str]): The prompt to pass. If prompt is not string
                then it will assume that chat mode / instruct mode is triggered.
            **kwargs (Optional[dict]): Optional keyword arguments.

        Additional Parameters:
            max_new_tokens (int): The maximum number of tokens to output. Defaults to 1024
            max_attention_window_size (int) Defaults to None
            sink_token_length (int): Defaults to None
            end_id (int): The end of sequence of ID of tokenize, defaults to tokenizer's default
                end id
            pad_id (int): The pd sequence of ID of tokenize, defaults to tokenizer's default end id
            temperature (float): The temperature to control probabilistic behaviour in generation
                Defaults to 1.0
            top_k (int): Defaults to 1
            top_p (float): Defaults to 1
            num_beams: (int): The number of responses to generate. Defaults to 1
            length_penalty (float): Defaults to 1.0
            repetition_penalty (float): Defaults to 1.0
            presence_penalty (float): Defaults to 0.0
            frequency_penalty (float): Defaults to 0.0
            early_stopping (int): Use this only when num_beams > 1, Defaults to 1
        """
        return self.request(prompt, **kwargs)
