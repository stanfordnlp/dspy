import dspy
from dspy.datasets import HotPotQA
from dsp.utils import deduplicate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import answer_exact_match, Evaluate
from dspy.teleprompt.finetune_teleprompter import convert_to_module_level_prompt_completion_data, bootstrap_data
import json

engine_args = {
            "enforce_eager": True,
            # "engine_use_ray": False,
            # "worker_use_ray": False,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1,
            "enable_lora": True
        }

NUM_THREADS = 70
lm = dspy.VLLMOfflineEngine3.instantiate_with_llm(model="meta-llama/Meta-Llama-3-8B-Instruct", engine_args=engine_args, batch_size=NUM_THREADS)
COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
rm = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)

dspy.settings.configure(lm=lm, rm=rm)

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

predict = BasicMH()

# print(predict(question="what is the capital of France?"))

dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)

trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

evaluate = Evaluate(devset=devset[:100], metric=answer_exact_match, num_threads=NUM_THREADS, display_progress=True)
evaluate(predict)