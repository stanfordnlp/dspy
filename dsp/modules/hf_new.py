
import os 
from typing import Optional, Union, Any 

from dsp.modules.lm import LM 
from dsp.utils.hf_utils import DeviceConfig

# TODO: Add extra bos token and custom token should be done while doing tokenization 
# TODO: After sometime, we also need to handle is_client case too

def get_kwargs(**kwargs):
    default_kwargs = {
        "parallelize": False,
        "trust_remote_code": True,
        "use_fast_tokenizer": True
    }
    return {**default_kwargs, **kwargs}

class HFLocalModel(LM):
    def __init__(
        self, 
        model: Union["transformers.PreTrainedModel", Any], 
        device: Optional[str]="auto",
        token: Optional[str]=None,
        revision: Optional[str] = "main",
        tokenizer: Optional[
            Union[
                str,
                "transformers.PreTrainedTokenizer",
                "transformers.PreTrainedTokenizerFast"
            ]
        ]=None,
        bnb_config: Optional["transformers.BitsAndBytesConfig"]=None, 
        **kwargs,
    ):
        try:
            import torch
            import transformers
            from dsp.utils.hf_utils import (
                setup_model_tokenizer_accelerator, _post_setup
            )
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install the following libraries: ",
                "transformers >= 4.30.0, torch==2.2.2, accelerator"
            ) from exc 
        
        
        model_name = model.name_or_path if not isinstance(model, str) else model
        super().__init__(model=model_name)

        # All the huggingface related instantiations 

        self.model_name = model_name
        self.provider = "hf"
        kwargs = get_kwargs(**kwargs)
        self.accelerator = None 
        
        if not isinstance(model, str):
            self.model, self.device, self.config = model, model.device, model.config 
            self.device_config = DeviceConfig(device=device, gpu_count=0)

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer 
            else:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=kwargs["trust_remote_code"],
                    use_fast=kwargs["use_fast_tokenizer"]
                )
        else:
            self.model, self.tokenizer, self.device_config, self.config, self.accelerator = setup_model_tokenizer_accelerator(
                model_name_or_path=model_name,
                token=token,
                bnb_quantization_config=bnb_config,
                **kwargs 
            )

        
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights() 
        
        if isinstance(model, str) and (self.device_config.gpu_count >= 1 or str(self.device_config.device)=="mps"):
            if not (kwargs.get("parallelize", None) or kwargs.get("autogptq", None) or hasattr(self, "accelerator") or bnb_config):
                try:
                    self.model.to(self.device_config.device)
                except ValueError:
                    print(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                    )

        self.model, self.accelerator, self.device_config = _post_setup(
            model_name_or_path=model_name, 
            model=self.model, 
            device_config=self.device_config, 
            parallelize=kwargs["parallelize"]
        )


        print("=> The model Loaded successfully")
    
    def basic_request(self, prompt):
        return prompt 
    
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return prompt 