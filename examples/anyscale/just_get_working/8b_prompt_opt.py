import dspy
from dspy.datasets import HotPotQA
from dsp.utils import deduplicate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import answer_exact_match, Evaluate
from dspy.teleprompt.finetune_teleprompter import convert_to_module_level_prompt_completion_data, bootstrap_data
import json

# TODO: increase max_tokens
lm = dspy.MultiOpenAI(model="lora2", api_base="http://localhost:8000/v1", api_key="sk-fake", api_provider="vllm")
COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
rm = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)

dspy.settings.configure(lm=lm, rm=rm)
# vllm serve meta-llama/Meta-Llama-3-8B-Instruct --pipeline_parallel_size=4 --enable_prefix_caching --enable_lora --lora-modules lora2=/home/ray/default/lora2
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

dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)

trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# sample_input = trainset[0]
# print(BasicMH().forward(**sample_input.inputs()))

OPTIMIZE=True

MAX_BOOTSTRAPPED_DEMOS = 3
MAX_LABELED_DEMOS = 3
NUM_CANDIDATE_PROGRAMS = 6
OPTIMIZER_NUM_TRAIN = 100
OPTIMIZER_NUM_VAL = 150

if OPTIMIZE:

    bfrs_optimizer = BootstrapFewShotWithRandomSearch(
        metric=answer_exact_match,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
        num_candidate_programs=NUM_CANDIDATE_PROGRAMS,
        num_threads=120
    )

    compiled_program = bfrs_optimizer.compile(BasicMH(), trainset=trainset[:OPTIMIZER_NUM_TRAIN], valset=trainset[OPTIMIZER_NUM_TRAIN:OPTIMIZER_NUM_VAL+OPTIMIZER_NUM_TRAIN])
    compiled_program.save(f"basicmh_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{OPTIMIZER_NUM_TRAIN}_{OPTIMIZER_NUM_VAL}_8bpft.json")
    # print(compiled_program)
else:
    compiled_program = BasicMH()
    compiled_program.load(f"basicmh_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{OPTIMIZER_NUM_TRAIN}_{OPTIMIZER_NUM_VAL}_8b.json")

START_IDX = 0
END_IDX = 1500
# evaluate = Evaluate(devset=devset[START_IDX:END_IDX], metric=answer_exact_match, num_threads=120, return_outputs=True, display_progress=True)
# score, results = evaluate(compiled_program)
# print(results)
bootstrapped_data = bootstrap_data(program=compiled_program, dataset=devset[START_IDX:END_IDX], metric=answer_exact_match, num_threads=120)
dataset = convert_to_module_level_prompt_completion_data(bootstrapped_data, program=compiled_program, exclude_demos=True)

filename = f"results_8bpft_{START_IDX}_{END_IDX}.json"
# Format the data for finetuning using the LM
print("Writing dataset with length", len(dataset), "to", filename)
with open(filename, "w") as f:
    json.dump(dataset, f)