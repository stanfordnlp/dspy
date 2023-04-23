from typing import Optional, Literal
from dsp.modules.lm import LM


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


class HFModel(LM):
    def __init__(self, model: str, checkpoint: Optional[str] = None, is_client: bool = False,
                 hf_device_map: Literal["auto", "balanced", "balanced_low_0", "sequential"] = "auto"):
        """wrapper for Hugging Face models

        Args:
            model (str): HF model identifier to load and use
            checkpoint (str, optional): load specific checkpoints of the model. Defaults to None.
            is_client (bool, optional): whether to access models via client. Defaults to False.
            hf_device_map (str, optional): HF config strategy to load the model. 
                Recommeded to use "auto", which will help loading large models using accelerate. Defaults to "auto".
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install Hugging Face transformers library to use HF models."
            ) from exc
        super().__init__(model)
        self.provider = "hf"
        self.is_client = is_client
        self.device_map = hf_device_map
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.is_client:
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model if checkpoint is None else checkpoint,
                    device_map=hf_device_map
                )
                self.drop_prompt_from_output = False
            except ValueError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model if checkpoint is None else checkpoint,
                    device_map=hf_device_map
                )
                self.drop_prompt_from_output = True
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.history = []

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def _generate(self, prompt, **kwargs):
        assert not self.is_client
        # TODO: Add caching
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [
            {"text": c}
            for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1:
            kwargs["num_beams"] = max(5, kwargs["n"])

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]


if __name__ == "__main__":
    model = HFModel(model="google/flan-t5-base")
    response = model("Who was the first man to walk on the moon?\nFinal answer: ")
    print(response)
