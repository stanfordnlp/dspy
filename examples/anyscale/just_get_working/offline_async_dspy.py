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

        self.basic_pred = BasicMH()
        self.thread_pool = ThreadPoolExecutor(max_workers=20)

    def __call__(self, inputs):
        results = {"results": []}
        futures: list[Future] = []

        for idx, question in enumerate(inputs["item"]):
            future = self.thread_pool.submit(self.process_question, question, idx)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results["results"].append(result)

        self.lm.shutdown()
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

def get_results(dataset, start_idx, end_idx):
    ds = ray.data.from_items([x.question for x in dataset[start_idx:end_idx]])
    concurrency = 1
    batch_size = math.ceil(ds.count() / concurrency)
    results = ds.map_batches(DSPyActor,
            batch_size=batch_size,
            num_gpus=1,
            concurrency=concurrency,
            fn_constructor_kwargs={"batch_size": batch_size}
        ).take_all()
    return results
    
def main(dataset):
    # ds = ray.data.from_items([f"What is 1 + {num}" for num in range(10000)])
    start_idx = 0
    end_idx = 10
    
    ds = ray.data.from_items([x.question for x in dataset[start_idx:end_idx]])
    LOAD_RESULTS = False

    if not LOAD_RESULTS:
        start = time.time()
        concurrency = 1
        count = ds.count()
        batch_size = math.ceil(count / concurrency)
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

    dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]
    main(devset)