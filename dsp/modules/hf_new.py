
from typing import Optional, Union, Any 

from dsp.modules.lm import LM 
from dsp.utils.hf_utils import (
    DeviceConfig, openai_to_hf, tokenize_and_encode, 
    setup_model_tokenizer_accelerator, _post_setup, model_generate, 
)

# TODO: GEMMA: Add extra bos token and custom token should be done while doing tokenization 
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
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.device, self.config = model, device, model.config 
            self.device_config = DeviceConfig(
                device=self.device, 
                torch_device=torch.device(self.device), 
                gpu_count=0, 
                
            )

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

        # TODO: Add history 
        print("=> The model Loaded successfully")
    

    def _generate(self, prompt: Union[str, list], **kwargs):
        # TODO: Need to add log_probs too 
        default_tokenizer_kwargs = {
            "left_truncate_len": None, 
            "add_special_tokens": None, 
            "add_bos_token": False 
        }

        default_generation_kwargs = {
            "temperature": 0.0, 
            "do_sample": False,  
            "max_length": None,
            "num_return_sequences": 1, 
            "repetition_penalty": 1.0, 
            "diversity_penalty": 0.0, 
            "max_new_tokens": 150
        }

        tokenizer_kwargs, generation_kwargs = dict(), dict()

        for k,v in default_tokenizer_kwargs.items():
            tokenizer_kwargs[k] = kwargs.get(k, v)
            if k in kwargs:
                kwargs.pop(k)
        stop=True if "stopping_criteria" in kwargs else False     

        # anything else provided will be treated as a generation kwargs 
        generation_kwargs = {
            **default_generation_kwargs, **openai_to_hf(**kwargs)
        }

        # now first tokenize
        # we will assume that if prompt is a dict then it is a chat mode

        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)

        input_ids = tokenize_and_encode(
            prompt, 
            config=self.config, 
            tokenizer=self.tokenizer, 
            **tokenizer_kwargs
        ).to(self.device_config.device)

        # Now get the output and decode 
        output_ids = model_generate(
            model=self.model, 
            tokenizer=self.tokenizer, 
            input_ids=input_ids,
            stop=stop, 
            **generation_kwargs
        )
        completions = [{"text": c} for c in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
        completion_dict = {
            "prompt": prompt, 
            "response": {
                "choices": completions, 
                "generation_kwargs": generation_kwargs, 
            },
            "kwargs": kwargs 
        }
        return completion_dict
    
    def basic_request(self, prompt: Union[str, list], **kwargs):
        response = self._generate(prompt, **kwargs)
        self.history.append(response)
        return response
    
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        
        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True
        
        response = self._generate(
            prompt=prompt,
            **kwargs
        )
        return [c["text"] for c in response["response"]["choices"]]