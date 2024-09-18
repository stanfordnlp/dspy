from llmforge.file_transfer import ModelDownloader
from llmforge.lora.utils import load_peft_model
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from google.cloud import storage
import os

def download_from_gcp_bucket(bucket_name, source_folder, destination_folder):
    """Downloads a folder from a GCP bucket to a local folder.

    Args:
        bucket_name (str): The name of the GCP bucket.
        source_folder (str): The path to the folder in the bucket.
        destination_folder (str): The local path where files should be saved.

    Returns:
        str: The path to the downloaded folder.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)

    for blob in blobs:
        if blob.name.endswith('/'):
            continue  # Skip directories
        
        relative_path = os.path.relpath(blob.name, source_folder)
        local_file_path = os.path.join(destination_folder, relative_path)
        
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        # only download if the file doesn't exist
        if not os.path.exists(local_file_path):
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob.name} to {local_file_path}")

    print(f"Folder {source_folder} downloaded to {destination_folder}.")
    return destination_folder

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# # lora_source_path = "gs://storage-bucket-cld-tffbxe9ia5phqr1unxhz4f7e1e/org_4snvy99zwbmh4gbtk64jfqggmj/cld_tffbxe9ia5phqr1unxhz4f7e1e/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:isaac:twzsb"
# lora_source_path = "gs://storage-bucket-cld-tffbxe9ia5phqr1unxhz4f7e1e/org_4snvy99zwbmh4gbtk64jfqggmj/cld_tffbxe9ia5phqr1unxhz4f7e1e/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:isaac:bdxcf"

# # Parse the GCS path
# bucket_name = lora_source_path.split('/')[2]
# source_folder = '/'.join(lora_source_path.split('/')[3:])

# # Download the LoRA model folder
# local_lora_path = download_from_gcp_bucket(bucket_name, source_folder, "/home/ray/default/lora2")
# print(local_lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_id) 
# peft_model = load_peft_model(lora_path=local_lora_path, base_ckpt_path=model_id, tokenizer_len=len(tokenizer), model_type="chat")

# llm = LLM(model=model_id, tokenizer=model_id, enable_lora=True)

# response = llm.generate(
#     prompts=["What is the capital of the moon?"],
#     sampling_params=SamplingParams(max_tokens=256),
#     lora_request=LoRARequest("tuned_lora_model", 1, local_lora_path)
# )

# print(response)
# ------------------------------------------------------------
import dsp
import dspy
import os
import dotenv
import traceback
import ray
import vllm
from vllm import LLM
from vllm.logger import init_logger
# import deploy_dspy
from dsp.deploy_dspy.async_llm import AsyncLLMWrapper
from dsp.deploy_dspy.download import download_model
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, Future
import time
import math
import json
from dsp.utils.utils import deduplicate
from dspy.datasets import HotPotQA
from dotenv import load_dotenv
logger = init_logger(__name__)
 #, "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(self.num_hops)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = []
        
        for hop in range(self.num_hops):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question).copy(context=context)
        return answer

class DSPyActor:
    def __init__(self, async_mode=False, batch_size=5):
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
        engine_args = {
            "max_pending_requests": 512,
            "enforce_eager": True,
            "engine_use_ray": False,
            "worker_use_ray": False,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1,
            "enable_lora": True
        }
        self.lm = dspy.VLLMOfflineEngine.instantiate_with_llm(model=model, engine_args=engine_args)
        COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
        self.rm = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)

        dspy.settings.configure(lm=self.lm, rm=self.rm, experimental=True)

        self.basic_pred = BasicMH()
        self.thread_pool = ThreadPoolExecutor(max_workers=30)

        # self.basic_pred.load("basicmh_8b_ft__3_3_6_100_100.json")


    def __call__(self, inputs):
        results = {"results": []}
        futures: list[Future] = []

        for idx, question in enumerate(inputs["item"]):
            future = self.thread_pool.submit(self.process_question, question, idx)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results["results"].append(result)

        return results

    def process_question(self, question, idx):
        if idx % 100 == 0:
            print(f"Processing question {idx}")
        # time.sleep(10 * idx)
        with dspy.context(lm=self.lm):
            try:
                pred = self.basic_pred(question=question)
                return {"question": question, "answer": pred.answer}
            except Exception as e:
                print("error", traceback.print_exception(e))
                return {"question": question, "error": str(e)}


def main(dataset):
    # ds = ray.data.from_items([f"What is 1 + {num}" for num in range(10000)])
    start_idx = 500
    end_idx = 1500

    
    ds = ray.data.from_items([x.question for x in dataset[start_idx:end_idx]])
    LOAD_RESULTS = False

    if not LOAD_RESULTS:
        start = time.time()
        concurrency = 4
        count = ds.count()
        batch_size = math.ceil(count / concurrency)
        results = ds.map_batches(DSPyActor,
                batch_size=batch_size,
                num_gpus=1,
                concurrency=concurrency,
                fn_constructor_kwargs={"batch_size": batch_size}
                ).take_all()
        end = time.time()
        print(f"Time taken: {end - start}")
        print(f"Total items: {count}")
        print(f"Time taken per item: {(end - start) / count}")
        with open(f"results_ft_8b_{start_idx}_{end_idx}.json", "w") as f:
            results = [result["results"] for result in results]
            json.dump(results, f)

    if LOAD_RESULTS:
        results = json.load(open(f"results_ft_8b_{start_idx}_{end_idx}.json", "r"))
    
    
    QA_map = {x.question: x for x in dataset[start_idx:end_idx]}
    predictions = [dspy.Prediction(answer=x["answer"], question=x["question"]) for x in results]
    metric = dspy.evaluate.answer_exact_match
    results = [metric(x, QA_map[x.question]) for x in predictions]
    print("results: ", sum(results), "/", len(results), f"= {sum(results) / len(results)}")

if __name__ == "__main__":
    # model_path = download_model("meta-llama/Meta-Llama-3-70B-Instruct")

    # import msgspec
    # print(isinstance(SamplingParams(), msgspec.Struct))
    # print(msgspec.structs.asdict(SamplingParams()))

    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"py_modules": [dspy, dsp, vllm], "env_vars": {"HF_HOME": "/mnt/local_storage/.cache/huggingface/", "HF_TOKEN": "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"}})

    dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]
    main(devset)