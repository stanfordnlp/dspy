# print(local_dir, model_id, lora_source_path)
# /mnt/local_storage/dspy/finetuning/Meta-Llama-3-8B-Instruct_isaac_pvslq meta-llama/Meta-Llama-3-8B-Instruct /mnt/local_storage/dspy/finetuning/Meta-Llama-3-8B-Instruct_isaac_pvslq

from typing import Dict
import dspy
import numpy as np
from llmforge.file_transfer import ModelDownloader
from llmforge.lora.utils import load_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import ray

local_dir = "/mnt/local_storage/dspy/finetuning/Meta-Llama-3-8B-Instruct_isaac_pvslq"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_source_path = local_dir

ray.init()

class DSPyActor:
    def __init__(self):

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        lora_source_path = local_dir
        downloader = ModelDownloader(model_id=model_id, source_path=lora_source_path)
        local_lora_path = downloader.download(tokenizer_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.peft_model = load_peft_model(lora_path=local_lora_path, base_ckpt_path=model_id, tokenizer_len=len(self.tokenizer), device="cuda")
        
        # self.program = BasicMH()
        # self.lm = HFProvidedModel(model=self.peft_model, tokenizer=self.tokenizer, max_tokens=251, pad_token_id=self.tokenizer.eos_token_id)

    def __call__(self, batch: Dict[str, np.ndarray]):
        print(batch)
        item = batch["data"][0]
        with dspy.context(lm=self.lm):
            # return {"output":self.program(item)}
            return {"output":item}
        

TEST_SIZE = 300
devset = np.random.randint(0, 50256, (1000, 1))
ds = ray.data.from_items(devset[:TEST_SIZE])
ds2 = ds.map_batches(
    DSPyActor,
    num_gpus=0.8,
    batch_size=1,
    concurrency=1,
)
print(ds2.take_batch(10))