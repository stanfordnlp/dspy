from tokenizers import AddedToken
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    def __init__(self, model, checkpoint=None, is_client=False):
        super().__init__(model)

        self.is_client = is_client
        if not self.is_client:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model if checkpoint is None else checkpoint).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.tokenizer.add_tokens(AddedToken("\n", normalized=False))

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
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, **kwargs)
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
