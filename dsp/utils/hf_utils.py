import os 
from packaging import version
from datetime import timedelta 
from typing import Optional, Literal, Union, Tuple, List 
from dataclasses import dataclass 


@dataclass 
class DeviceConfig:
    device: str
    torch_device: Optional["torch.device"] = None 
    gpu_count: int = 0
    rank: Optional[int] = None 
    world_size: Optional[int] = None

def get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""   
    # Reference: # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py 

    import torch 

    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args

def get_model_type(
    config: Union["transformers.PretrainedConfig", "transformers.AutoConfig"],
    backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
) -> "transformers.PreTrainedModel":
    assert backend in ["default", "causal", "seq2seq"], ValueError(
        "backend can be eihter of the following: ",
        "'default', 'causal', 'seq2seq'"
    )

    import transformers 
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    )

    if backend == "default":
        if (getattr(config, "model_type")) in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            model_type = transformers.AutoModelForSeq2SeqLM
        elif (getattr(config, "model_type")) in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            model_type = transformers.AutoModelForCausalLM
        else:
            print("Defaulting to Causal LM type")
            model_type = transformers.AutoModelForCausalLM
    else:
        if backend == "causal":
            model_type = transformers.AutoModelForCausalLM
        else:
            model_type = transformers.AutoModelForSeq2SeqLM
    return model_type 

def setup_device(device: Optional[str]=None, parallelize: Optional[bool]=False) -> Tuple["torch.device", DeviceConfig, "Accelerator"]:
    import torch 
    from accelerate import Accelerator, InitProcessGroupKwargs
    gpus = torch.cuda.device_count()
    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

    if not (parallelize or accelerator.num_processes > 1):
        device_set = set(
            ["cuda", "cpu"]
            + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            + ["mps", "mps:0"]
        )

        if device in device_set:
            device_config = DeviceConfig(device=device)
            torch_device = torch.device(device)

            if device in ("mps", "mps:0") and version.parse(torch.__version__) < version.parse("2.1"):
                raise RuntimeError(
                    f"mps requires torch >= 2.1. You have {torch.__version__}"
                )
            print(f"Using device: '{str(device)}'")
        else:
            print("Device is not specified")
            device_ = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu") 
            device_config = DeviceConfig(device=device_)
            torch_device = torch.device(device_)
    else:
        if device != "!cuda":
            print(
                f"Using `accelerate launch` or `parallelize=True`, device '{device}' ", "will be overridden when placing model."
            )
        torch_device = torch.device(device)
        device_config = DeviceConfig(device=device)
    
    device_config.gpu_count = gpus
    device_config.torch_device = torch_device
    return device_config, accelerator

def get_dtype(dtype: Union[str, "torch.dtype"]) -> "torch.dtype":
    # reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/utils.py#L198
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    import torch 

    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

def setup_tokenizer(
    model_name_or_path: str,
    config: "transformers.AutoConfig",
    revision: Optional[str]="main",
    trust_remote_code: Optional[bool] = False,
    use_fast_tokenizer: Optional[bool] = True,
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    
    import transformers 

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer, 
    )

    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if getattr(config, "model_type", None) == "qwen":
            tokenizer.pad_token = "<|endoftext|>"
        elif tokenizer.__class__.__name__ in ["RWKVWorldTokenizer", "Rwkv5Tokenizer"]:
            assert tokenizer.pad_token_id == 0
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer 

def parse_model_kwargs(**kwargs):
    default_kwargs = {
        "revision": "main",
        "dtype": "auto",
        "parallelize": False,
        "trust_remote_code": False,
        "device_map_option": "auto",
        "max_memory_per_gpu": None,
        "max_cpu_memory": None,
        "offload_folder": "./offload",
        "peft": None,
        "delta": None,
        "autogptq": False 
    }

    return {**default_kwargs, **kwargs} 

def _either_keep_bnbconfig_or_kwargs(bnb_config: Optional["transformers.BitsAndBytesConfig"], **kwargs):
    # Assumption: this will never be None since, we merge it with default paramters
    
    if bnb_config is None:
        return kwargs 
    else:
        keys_to_pop = [
            "use_4bit", 
            "bnb_4bit_compute_dtype", 
            "bnb_4bit_quant_type", 
            "use_nested_quant",
            "load_in_8bit",
            "load_in_4bit",
            "llm_int8_threshold",
            "llm_int8_skip_modules",
            "llm_int8_enable_fp32_cpu_offload",
            "llm_int8_has_fp16_weight",
            "bnb_4bit_use_double_quant",
            "bnb_4bit_quant_storage"
        ]

        for key in keys_to_pop:
            if key in kwargs:
                kwargs.pop(key)
        return kwargs 

def setup_model(
    model_name_or_path: str,
    config: "transformers.AutoConfig",
    accelerator: Optional["Accelerator"]=None, 
    device: Optional[str]="cuda",
    backend: Optional[str]="causal",
    # model kwargs starts from here (for get_model_from_name_or_path function)
    token: Optional[str]=None, 
    revision: Optional[str] = "main",
    dtype: Optional[Union[str, "torch.dtype"]] = "auto",
    trust_remote_code: Optional[bool] = False,
    parallelize: Optional[bool] = False,
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
    peft: Optional[str] = None,
    delta: Optional[str] = None,
    quantization_config: Optional["transformers.BitsAndBytesConfig"]=None, 
    autogptq: Optional[Union[bool, str]] = False,
    **kwargs,
) -> "transformers.PreTrainedModel":
    
    import transformers

    model_kwargs = kwargs if kwargs else {}
    if parallelize:
        model_kwargs.update(
            get_accelerate_args(
                device_map_option, 
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )
        )
    elif "device_map" not in model_kwargs:
        if accelerator is not None:
            model_kwargs.update(
                {"device_map": {"": f"cuda:{accelerator.local_process_index}"}}
            )
        else:
            model_kwargs.update(
                model_kwargs.update({"device_map": {"": str(device)}})
            )
    
    # Loading model with bitsandbytes quantization
    model_kwargs = _either_keep_bnbconfig_or_kwargs(bnb_config=quantization_config, **model_kwargs)


    if not autogptq:
        if quantization_config is not None or model_kwargs.get("load_in_4bit", None):
            assert transformers.__version__ >= "4.30.0", ValueError("load_in_4bit requires transformers >= 4.30.0")
        
        model_class = get_model_type(config=config, backend=backend)
        
        if quantization_config is None:
            if model_kwargs.get("load_in_4bit", None):
                if model_kwargs.get("bnb_4bit_compute_dtype", None):
                    model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(model_kwargs["bnb_4bit_compute_dtype"])
            model = model_class.from_pretrained(
                model_name_or_path,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                token=dict(token=token or os.environ.get("HF_TOKEN")),
                **model_kwargs 
            )
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                token=dict(token=token or os.environ.get("HF_TOKEN")),
                quantization_config=quantization_config,
                **model_kwargs 
            )
    
    else:
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ModuleNotFoundError:
            raise Exception(
                "Please install HF AutoGPTQ using this command: ", 
                "pip install auto-gptq"
            )
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            model_basename=None if autogptq is True else Path(autogptq).stem,
            use_safetensors=True if autogptq is True else autogptq.endswith(".safetensors"),
            token=dict(token=token or os.environ.get("HF_TOKEN"))
            **model_kwargs
        )
    
    # Setup PEFT 
    if peft and delta:
        raise ValueError(
            "Cannot use both 'peft' and 'delta' options at the same time."
    )
    
    if peft:
        try:
            from peft import PeftModel
            from peft import __version__ as peft_version
        except ModuleNotFoundError:
            raise Exception(
                "Please install HF AutoGPTQ using this command: ", 
                "pip install peft"
            )

        if model_kwargs.get("load_in_4bit", None) or quantization_config is not None:
            if version.parse(peft_version) < version.parse("0.4.0"):
                raise AssertionError("load_in_4bit requires peft >= 0.4.0") 
        
        model = PeftModel.from_pretrained(model, peft, revision=revision)
    elif delta:
        if autogptq:
            print("Delta weights might trigger unexpected behavior when used with AutoGPTQ.")
        
        model_delta = model_class.from_pretrained(
            delta,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs
        )

        for name, param in model.state_dict().items():
            try:
                param.delta += model_delta.state_dict().items() 
            except KeyError:
                raise KeyError(f"Delta model is missing weights for layer: {name}")
            except Exception as e:
                 raise RuntimeError(f"Failed to add delta weights to layer {name}. Error: {e}")
        del model_delta 
    return model 

def setup_model_tokenizer_accelerator(
    model_name_or_path: str,
    parallelize: Optional[bool]=False,
    backend: Optional[str] = "causal", 
    device: Optional[str] = "cuda",
    revision: Optional[str] = "main",
    trust_remote_code: Optional[bool] = False,
    use_fast_tokenizer: Optional[bool]=False, 
    bnb_quantization_config: Optional["transformers.BitsAndBytesConfig"]=None, 
    **kwargs
):
    import transformers 

    device_config, accelerator = setup_device(
        device=device, parallelize=parallelize
    )

    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path, revision=revision, trust_remote_code=trust_remote_code
    )

    kwargs_to_pass = parse_model_kwargs(**{
        **kwargs, 
        "parallelize": parallelize,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    })

    model = setup_model(
        model_name_or_path=model_name_or_path,
        config=config,
        device=device_config.torch_device,
        accelerator=accelerator,
        backend=backend,
        quantization_config=bnb_quantization_config,
        **kwargs_to_pass
    )

    tokenizer = setup_tokenizer(
        model_name_or_path=model_name_or_path,
        config=config,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer
    )
    return model, tokenizer, device_config, config, accelerator

# Due to lack of multi GPU setup this is not been tested fully 
# source of this code is taken from here: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py

def _post_setup(
    model_name_or_path: Union[str, "transformers.PreTrainedModel"], 
    model: "transformers.PreTrainedModel", 
    device_config: DeviceConfig,
    parallelize: bool, 
    accelerator: Optional["Accelerator"]=None,  
):  
    import torch 
    from accelerate import DistributedType

    if isinstance(model_name_or_path, str):
        if device_config.gpu_count > 1 and accelerator is not None:
            if parallelize:
                if accelerator.num_processes > 1:
                    raise RuntimeError(
                        "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                    )
                else:
                    pass 
            elif accelerator.num_processes == 1:
                device_config.rank = 0
                device_config.world_size = 1 
            else:
                if device_config.gpu_count > accelerator.num_processes:
                    print(
                        "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                        "If you would like to use data parallelism, please launch the script "
                        "with 'accelerate launch *script*'. "
                        f"Current run will proceed with {accelerator.num_processes} devices."
                    )
                assert (accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_CPU]), ValueError(
                    "Unsupported distributed type provided. Only DDP and FSDP are supported."
                )

                if accelerator.distributed_type == DistributedType.FSDP:
                    model = accelerator.prepare(accelerator.unwrap_model(model), evaluation_mode=True)
                else:
                    model = accelerator.prepare_model(accelerator.unwrap_model(model), evaluation_mode=True)
                device_config.torch_device = torch.device(f"cuda:{accelerator.local_process_index}")

                if accelerator.is_local_main_process:
                    print(f"Using {device_config.gpu_count} devices with data parallelism")
                device_config.rank = accelerator.local_process_index
                device_config.world_size = accelerator.num_processes
    else:
        print("Passed an already initialized model through pre-trained")
        device_config.rank = 0
        device_config.world_size = 1 
    return model, accelerator, device_config


## Here comes all the encoding and decoding part for huggingface 
# Assumption: Right now dspy.LM.__call__ method only takes one prompt so batching is not supported
# in this version 

def tokenize_and_encode(
    prompt: str, config: "transformers.AutoConfig", 
    tokenizer: "transformers.AutoTokenizer", 
    left_truncate_len: Optional[int]=None, 
    add_special_tokens:Optional[Union[str, bool]]=None, 
    add_bos_token: Optional[int]=False
) -> "torch.tensor":
    
    import torch 
    import transformers 
    special_tokens_kwargs = {}
    model_class = get_model_type(config=config, backend="default")
    
    if add_special_tokens is None:
        if model_class == transformers.AutoModelForCausalLM:
            special_tokens_kwargs = {
                "add_special_tokens": False or add_bos_token
            }
    else:
        special_tokens_kwargs = {"add_special_tokens": add_special_tokens}
    encoding = tokenizer.encode(prompt, **special_tokens_kwargs)
    
    # left-truncate the encoded context to be at most `left_truncate_len` tokens long
    if left_truncate_len:
        encoding = encoding[-left_truncate_len:]
    encoding = torch.tensor([encoding])
    return encoding 

def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


def stop_sequences_criteria(
    tokenizer: "transformers.PreTrainedTokenizer",
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> "transformers.StoppingCriteriaList":
    # Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py 
    import transformers 

    class MultiTokenEOSCriteria(transformers.StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""
        # Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/utils.py 

        def __init__(
            self,
            sequence: str,
            tokenizer: "transformers.PreTrainedTokenizer",
            initial_decoder_input_length: int,
            batch_size: int,
        ) -> None:
            self.initial_decoder_input_length = initial_decoder_input_length
            self.done_tracker = [False] * batch_size
            self.sequence = sequence
            self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
            # print(sequence, self.sequence_ids)
            # we look back for 2 more tokens than it takes to encode our stop sequence
            # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
            # and we don't want to mistakenly not stop a generation because our
            # (string) stop sequence was output in a different tokenization

            # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
            # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
            # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
            self.sequence_id_len = len(self.sequence_ids) + 2
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
            lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

            lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

            for i, done in enumerate(self.done_tracker):
                if not done:
                    self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
            return False not in self.done_tracker


    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

def model_generate(
    model: "transformers.PreTrainedModel", 
    tokenizer: "transformers.PreTrainedTokenizer", 
    input_ids: "torch.tensor",  
    stop: Optional[List[str]]=None, 
    **generation_kwargs,
):  
    # This helps to remove all the warnings 

    generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
    do_sample = generation_kwargs.get("do_sample", None)

    if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
        generation_kwargs["do_sample"] = do_sample = False
    
    if do_sample == True or generation_kwargs.get("temperature") == 0.0:
        # otherwise it throws error 
        generation_kwargs["temperature"] = generation_kwargs["temperature"] + 0.1
 
    if stop:
        # For our case, batch size will always be one 
        stopping_criteria = stop_sequences_criteria(
            tokenizer, 
            stop, 
            input_ids.shape[1], 
            input_ids.shape[0]
        )
        generation_kwargs["stopping_criteria"] = stopping_criteria

    return model.generate(
        input_ids=input_ids, **generation_kwargs
    )