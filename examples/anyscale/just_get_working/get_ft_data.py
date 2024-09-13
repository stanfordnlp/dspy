import os
x
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

if ray.is_initialized():
    ray.shutdown()
ray.init(runtime_env={"py_modules": [dspy, dsp], "env_vars": {"HF_HOME": "/mnt/local_storage/.cache/huggingface/"}})

TRAIN_SIZE = 10
EVAL_SIZE = int(TRAIN_SIZE/4)

hotpot_dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, keep_details="type")
trainset = [x.with_inputs('question') for x in hotpot_dataset.train]
devset = [x.with_inputs('question') for x in hotpot_dataset.dev]

# Set up metrics
NUM_THREADS = 100

colbert_v2_endpoint = "http://20.102.90.50:2017/wiki17_abstracts"
colbertv2 = dspy.ColBERTv2(url=colbert_v2_endpoint)

engine_args = {
    "max_pending_requests": 512,
    "enforce_eager": True,
    "engine_use_ray": False,
    "worker_use_ray": False,
    "enable_prefix_caching": False,
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.95,
    "max_model_len": 40960,
}
lm = VLLMOfflineEngine.instantiate_with_llm(model="meta-llama/Meta-Llama-3.1-70B-Instruct", engine_args=engine_args)

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

if COMPILE:
    num_train, num_dev = 100, 100
    max_bootstrapped_demos, max_labeled_demos, num_candidate_programs = 3,3,6
    config = dict(max_bootstrapped_demos=max_bootstrapped_demos, num_candidate_programs=num_candidate_programs, num_threads=NUM_THREADS)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, **config)
    basicmh_bs = teleprompter.compile(BasicMH(), trainset=trainset[:num_train], valset=devset[:num_dev])
    basicmh_bs.save(f"basicmh_70b_{max_bootstrapped_demos}_{max_labeled_demos}_{num_candidate_programs}_{num_train}_{num_dev}.json")

    # baseline_eval = evaluate(BasicMH(), devset=devset[:300])
else:
    basicmh_bs = BasicMH()
    basicmh_bs.load("basicmh_70b_3_3_6_100_100.json")
    # bs_eval = evaluate(basicmh_bs, devset=devset[:10])

# exit()
dc_kwargs = {
    "sampling_temperature": 0.9,
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
    temp_trainset = data.copy()
    sampling_temperature = 0.9
    sampling_temperature_delta = 0.0001
    for i in range(3):
        NUM_THREADS = 100
        print(f"Bootstrapping round {i}")
        checkpoint_file = f"checkpoint_{filename.split('.')[0]}_{len(data)}_{len(temp_trainset)}_{i}.json"
        
        try:
            # Split the bootstrapping process into 4 checkpoints
            for checkpoint in range(2):
                mini_checkpoint_file = f"mini_checkpoint_{filename.split('.')[0]}_{len(data)}_{len(temp_trainset)}_{i}_checkpoint_{checkpoint}.json"
                if os.path.exists(mini_checkpoint_file):
                    print(f"Skipping checkpoint {checkpoint} for round {i} because it already exists")
                    with open(mini_checkpoint_file, 'r') as f:
                        correct_questions = [d["prompt"].split("Question: ")[2].split("\n")[0].strip() for d in json.load(f)]
                        temp_trainset = [d for d in temp_trainset if d["question"] not in correct_questions]
                    print(f"Temp trainset length after bootstrapping round {i}, checkpoint {checkpoint}: {len(temp_trainset)}")
                    continue
                chunk_size = len(temp_trainset) // 4
                start_idx = checkpoint * chunk_size
                end_idx = start_idx + chunk_size if checkpoint < 3 else len(temp_trainset)
                
                chunk_trainset = temp_trainset[start_idx:end_idx]
                
                bootstrapped_chunk = bootstrap_data_for_round(program=basicmh_bs, dataset=chunk_trainset, metric=metric, num_threads=NUM_THREADS, sampling_round=i, sampling_temperature=sampling_temperature, sampling_temperature_delta=sampling_temperature_delta)
                print(f"Bootstrapped data for round {i}, checkpoint {checkpoint}:", bootstrapped_chunk)
                print(f"Scores for round {i}, checkpoint {checkpoint}:", [(d["score"], d["example"]) for d in bootstrapped_chunk])
                
                # Post process the data to remove any entries with no score
                filtered_chunk = [d for d in bootstrapped_chunk if d["score"]]
                questions = [d["example"]["question"] for d in filtered_chunk]
                print(f"Filtered data for round {i}, checkpoint {checkpoint}:", filtered_chunk)

                # Save checkpoint
                with open(mini_checkpoint_file, 'w') as f:
                    dataset = convert_to_module_level_prompt_completion_data(filtered_chunk, program=basicmh_bs, exclude_demos=True)
                    json.dump(dataset, f)
                
                correct_data.extend(dataset)
                temp_trainset = [d for d in temp_trainset if d["question"] not in questions]
                print(f"Temp trainset length after bootstrapping round {i}, checkpoint {checkpoint}: {len(temp_trainset)}")
                
            sampling_temperature += sampling_temperature_delta
        
        except Exception as e:
            print(f"Error in round {i}: {str(e)}")
            # If there's an error, try to load the last successful checkpoint
            if i > 0:
                prev_checkpoint = f"checkpoint_{filename.split('.')[0]}_{len(data)}_{len(temp_trainset)}_{i-1}.json"
                if os.path.exists(prev_checkpoint):
                    with open(prev_checkpoint, 'r') as f:
                        correct_data = json.load(f)
                    print(f"Loaded data from previous checkpoint: {prev_checkpoint}")
                break
            else:
                raise e

    # After all rounds, combine checkpoints if we made it to the end
    if i == 2:  # Assuming 3 rounds (0, 1, 2)
        for round_num in range(3):
            re = 
            checkpoint_file = f"checkpoint_{filename.split('.')[0]}_{len(data)}_{len(temp_trainset)}_{round_num}.json"
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    correct_data.extend(json.load(f))
        print("Combined all checkpoints successfully")
        
    # Convert the data to prompt completion format
    dataset = convert_to_module_level_prompt_completion_data(correct_data, program=basicmh_bs, exclude_demos=True)
    
    # Format the data for finetuning using the LM
    print("Writing dataset with length", len(dataset), "to", filename)
    with open(filename, "w") as f:
        ujson.dump(dataset, f)

for key, data in dataset_filenames.items():
    write_data(data, key)

