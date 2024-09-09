import dsp
import dspy
import os
import dotenv
import traceback
import ray
import vllm
from vllm import LLM
from vllm.logger import init_logger
import deploy_dspy
from deploy_dspy.async_llm import AsyncLLMWrapper
from deploy_dspy.download import download_model
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, Future
import time
import math
import json
from dsp.utils.utils import deduplicate
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dotenv import load_dotenv
logger = init_logger(__name__)
 #, "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

class DSPyActor:
    def __init__(self, async_mode=False, batch_size=5):
        dotenv.load_dotenv()
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
        engine_args = {
            "max_pending_requests": 512,
            "enforce_eager": True,
            "engine_use_ray": False,
            "worker_use_ray": False,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1
        }
        self.lm = dspy.VLLMOfflineEngine.instantiate_with_llm(model=model, engine_args=engine_args)
        COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
        self.rm = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)

        dspy.settings.configure(lm=self.lm, rm=self.rm, experimental=True)

        self.basic_pred = CoT()
        self.thread_pool = ThreadPoolExecutor(max_workers=512)

    def __call__(self, inputs):
        results = {"results": []}
        futures: list[Future] = []

        print(len(inputs["item"]))
        for idx, question in enumerate(inputs["item"]):
            future = self.thread_pool.submit(self.process_question, question, idx)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results["results"].append(result)

        # self.lm.shutdown()
        return results

    def process_question(self, question, idx):
        # time.sleep(10 * idx)
        with dspy.context(lm=self.lm):
            try:
                pred = self.basic_pred(question=question)
                return {"question": question, "answer": pred.answer}
            except Exception as e:
                print("error", traceback.print_exception(e))
                return {"question": question, "error": str(e)}
    
def main(dataset):
    start_idx = 0
    end_idx = 2
    # ds = ray.data.from_items([f"What is 1 + {num}" for num in range(10000)])
    
    ds = ray.data.from_items([x.question for x in dataset[start_idx:end_idx]])
    LOAD_RESULTS = False

    if not LOAD_RESULTS:
        start = time.time()
        concurrency = 1
        count = ds.count()
        batch_size = math.ceil(count / concurrency)
        print(f"Batch size: {batch_size}")
        results = ds.map_batches(DSPyActor,
                batch_size=batch_size,
                num_gpus=1,
                concurrency=concurrency,
                fn_constructor_kwargs={"batch_size": batch_size}
            ).take_all()

        end = time.time()
        print("results: ", results)
        print(f"Time taken: {end - start}")
        print(f"Total items: {count}")
        print(f"Time taken per item: {(end - start) / count}")
    #     with open(f"results_8b_{start_idx}_{end_idx}.json", "w") as f:
    #         # results is of form [{"results": {"question": ..., "answer": ...}}]
    #         # we need to flatten it to [{"question": ..., "answer": ...}]
    #         results = [result["results"] for result in results]
    #         json.dump(results, f)

    # if LOAD_RESULTS:
    #     results = json.load(open(f"results_8b_{start_idx}_{end_idx}.json", "r"))
    
    
    # QA_map = {x.question: x for x in dataset[start_idx:end_idx]}
    # predictions = [dspy.Prediction(answer=x["answer"], question=x["question"]) for x in results]
    # metric = dspy.evaluate.answer_exact_match
    # results = [metric(x, QA_map[x.question]) for x in predictions]
    # print("results: ", sum(results), "/", len(results), f"= {sum(results) / len(results)}")

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"py_modules": [dspy, dsp, deploy_dspy], "env_vars": {"HF_HOME": "/mnt/local_storage/.cache/huggingface/"}})

    gms8k = GSM8K()

    trainset, devset = gms8k.train, gms8k.dev

    main(trainset)