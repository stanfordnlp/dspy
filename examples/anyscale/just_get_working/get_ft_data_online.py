import os
import ray
import dspy
import dsp
import dspy.evaluate
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from dotenv import load_dotenv
from dsp.utils.utils import deduplicate
from concurrent.futures import Future
import time
from typing import Any, List, Optional, Literal, Union
import ujson
import openai
from dsp.modules.lm import TrainableLM, TrainingMethod
from dsp.modules.trainable_anyscale import TrainableAnyscale
from dsp.modules.vllm_engine import VLLMOfflineEngine
from dspy.teleprompt.finetune_teleprompter import bootstrap_data
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.finetune_teleprompter import convert_to_module_level_prompt_completion_data
from dspy.teleprompt.finetune_teleprompter import bootstrap_data_for_round
from collections import defaultdict
import yaml
import json
load_dotenv()
# Altenatively, you can set the environment variables in code
os.environ["DSP_CACHEDIR"] = "/mnt/cluster_storage/dspy/cache"
os.environ["HF_HOME"] = "/mnt/local_storage/.cache/huggingface/"

# if ray.is_initialized():
#     ray.shutdown()
# ray.init(runtime_env={"py_modules": [dspy, dsp], "env_vars": {"HF_HOME": "/mnt/local_storage/.cache/huggingface/"}})

TRAIN_SIZE = 10
EVAL_SIZE = int(TRAIN_SIZE/4)

hotpot_dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, keep_details="type")
trainset = [x.with_inputs('question') for x in hotpot_dataset.train]
devset = [x.with_inputs('question') for x in hotpot_dataset.dev]

# Set up metrics
NUM_THREADS = 90

colbert_v2_endpoint = "http://20.102.90.50:2017/wiki17_abstracts"
colbertv2 = dspy.ColBERTv2(url=colbert_v2_endpoint)

normal_llama = "meta-llama/Meta-Llama-3-70B-Instruct"
lm = dspy.MultiOpenAI(
        model=normal_llama,
        api_base="https://endpoints-v2-enough-gray-iron-mglb8.cld-tffbxe9ia5phqr1u.s.anyscaleuserdata.com/v1",
        api_key="sk-anyscale-api-key",
        api_provider="vllm",
        model_type="chat",
    )
dspy.settings.configure(rm=colbertv2, lm=lm, experimental=True)   

# x = dspy.Predict("question -> answer")
# print(x(question="How many people live in the world?"))

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question, return_trace=False):
        context = []
        for hop in range(2):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)

        x = self.generate_answer(context=context, question=question).copy(context=context)
        
        if return_trace:
            return x, dspy.settings.trace
        return x

COMPILE = False

metric = dspy.evaluate.answer_exact_match
kwargs = dict(num_threads=NUM_THREADS, display_progress=True)
evaluate = Evaluate(devset=devset, metric=metric, **kwargs)

MAX_BOOTSTRAPPED_DEMOS = 3
MAX_LABELED_DEMOS = 3
NUM_CANDIDATE_PROGRAMS = 6
OPTIMIZER_NUM_TRAIN = 100
OPTIMIZER_NUM_VAL = 150

if COMPILE:
    config = dict(max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS, num_candidate_programs=NUM_CANDIDATE_PROGRAMS, num_threads=NUM_THREADS)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, **config)
    basicmh_bs = teleprompter.compile(BasicMH(), trainset=trainset[:OPTIMIZER_NUM_TRAIN], valset=devset[:OPTIMIZER_NUM_VAL])
    basicmh_bs.save(f"basicmh_70b_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{OPTIMIZER_NUM_TRAIN}_{OPTIMIZER_NUM_VAL}.json")

    # baseline_eval = evaluate(BasicMH(), devset=devset[:300])
else:
    basicmh_bs = BasicMH()
    basicmh_bs.load(f"basicmh_70b_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{OPTIMIZER_NUM_TRAIN}_{OPTIMIZER_NUM_VAL}.json")
    # bs_eval = evaluate(basicmh_bs, devset=devset[:10])

# exit()
dc_kwargs = {
    "sampling_temperature": 0.92,
    "sampling_temperature_delta":0.0001,
    "num_threads": NUM_THREADS,
}
TRAIN_SIZE = 1500
EVAL_SIZE = int(TRAIN_SIZE/4)

dataset_filenames = {f"trainset_data_{TRAIN_SIZE}.jsonl": trainset[:TRAIN_SIZE], f"devset_data_{EVAL_SIZE}.jsonl": trainset[TRAIN_SIZE:TRAIN_SIZE+EVAL_SIZE]}

def write_data(data, filename):
    # get the bootstrapped data for num_rounds=1, but using the callback
    print("Bootstrapping and writing data to", filename)
    correct_data = []
    unsolved_examples = data.copy()
    sampling_temperature = 0.9
    sampling_temperature_delta = 0.0001
    
    for i in range(3):
        if len(unsolved_examples) == 0:
            break
        data = bootstrap_data_for_round(basicmh_bs, unsolved_examples, metric=metric, num_threads=NUM_THREADS, sampling_round=i, sampling_temperature=sampling_temperature, sampling_temperature_delta=sampling_temperature_delta)
        correct_data_round = [x for x in data if x["score"]]
        correct_examples_round = set([x["example"] for x in correct_data_round])
        unsolved_examples = [x for x in unsolved_examples if x not in correct_examples_round]
        correct_data.extend(correct_data_round)
        sampling_temperature += sampling_temperature_delta
    
    # Convert the data to prompt completion format
    dataset = convert_to_module_level_prompt_completion_data(correct_data, program=basicmh_bs, exclude_demos=True)
    
    # Format the data for finetuning using the LM
    print("Writing dataset with length", len(dataset), "to", filename)
    with open(filename, "w") as f:
        ujson.dump(dataset, f)

for key, data in dataset_filenames.items():
    write_data(data, key)

