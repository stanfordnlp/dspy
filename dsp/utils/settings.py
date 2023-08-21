from typing import TypedDict, Optional
from contextlib import contextmanager
from dsp.utils.utils import dotdict

class CompilerConfig(TypedDict):
    model_name: str
    new_model: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    use_nested_quant: bool
    output_dir: str
    num_train_epochs: int
    fp16: bool
    bf16: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    auto_find_batch_size: bool
    max_grad_norm: float
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    save_steps: int
    logging_steps: int
    max_seq_length: Optional[int]
    packing: bool
    device_map: str

def get_default_compiler_config() -> CompilerConfig:
    return {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "new_model": "Llama-2-7b-chat-hf-ft",
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "use_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "use_nested_quant": False,
        "output_dir": "./results",
        "num_train_epochs": 1,
        "fp16": False,
        "bf16": False,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "auto_find_batch_size": True,
        "max_grad_norm": 0.3,
        "learning_rate": 2e-4,
        "weight_decay": 0.001,
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": "constant",
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "group_by_length": True,
        "save_steps": 25,
        "logging_steps": 25,
        "max_seq_length": None,
        "packing": False,
        "device_map": "auto"
    }

class Settings(object):
    """DSP configuration settings."""

    _instance = None
    branch_idx: int = 0

    def __new__(cls):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = []

            #  TODO: remove first-class support for re-ranker and potentially combine with RM to form a pipeline of sorts
            #  eg: RetrieveThenRerankPipeline(RetrievalModel, Reranker)
            #  downstream operations like dsp.retrieve would use configs from the defined pipeline.
            config = dotdict(
                lm=None,
                rm=None,
                reranker=None,
                compiled_lm=None,
                force_reuse_cached_compilation=False,
                compiling=False,
                compiler_config=get_default_compiler_config(),
            )
            cls._instance.__append(config)

        return cls._instance

    @property
    def config(self):
        return self.stack[-1]

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)

        if name in self.config:
            return self.config[name]

        super().__getattr__(name)

    def __append(self, config):
        self.stack.append(config)

    def __pop(self):
        self.stack.pop()

    def configure(self, inherit_config: bool = True, **kwargs):
        """Set configuration settings.

        Args:
            inherit_config (bool, optional): Set configurations for the given, and use existing configurations for the rest. Defaults to True.
        """
        if inherit_config:
            config = {**self.config, **kwargs}
            for k, v in kwargs.items():
                if k == 'compiler_config':
                    if not isinstance(v, dict):
                        raise ValueError('compiler_config must be a dict')

                    config[k] = {**get_default_compiler_config(), **v}
                else:
                    config[k] = v
        else:
            config = {**kwargs}

        self.__append(config)

    @contextmanager
    def context(self, inherit_config=True, **kwargs):
        self.configure(inherit_config=inherit_config, **kwargs)

        try:
            yield
        finally:
            self.__pop()

    def __repr__(self) -> str:
        return repr(self.config)


settings = Settings()
