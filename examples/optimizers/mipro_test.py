import dspy
import os; 
os.environ['LITELLM_LOG'] = 'INFO'
from datetime import datetime

# lm = dspy.LM("gpt-4o-mini-2024-07-18", api_base=os.getenv('OPENAI_API_BASE'), api_key=os.getenv('OPENAI_API_KEY'))
# lm = dspy.HFClientVLLM(
#     model="meta-llama/Meta-Llama-3-8B",
#     port=7410,
#     url=["http://future-hgx-1:7501"],
#     max_tokens=250,
#     stop=["\n\n", "\n---", "assistant"],
# )

# lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B", api_base="http://future-hgx-1:7501/v1", api_key="EMPTY", temperature=0.4, max_tokens=500)
lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B", api_base="http://future-hgx-1:7501/v1", api_key="EMPTY", max_tokens=700)
# lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B", api_base="http://future-hgx-1:7501/v1", api_key="EMPTY")


dspy.configure(lm=lm)

# dspy.ChainOfThought('question -> answer: int')(question="What is two plus two?")
# breakpoint()

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

gms8k = GSM8K()

trainset, devset = gms8k.train, gms8k.dev

program = dspy.ChainOfThought('question -> answer: int')

from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:100], metric=gsm8k_metric, num_threads=1, display_progress=True, display_table=False, max_errors=10000)

# class GSM8Reflection(dspy.Module):
#     def __init__(self, n = 3):
#         super().__init__()
#         self.gen_candidate_answer = dspy.ChainOfThought("question -> answer:int", n=n)
#         self.gen_final_answer = dspy.ChainOfThought("candidate_answers -> final_answer:int")
    
#     def get_candidate_answers_string(self, candidate_answers):
#         return "\n".join([f"Potential Answer {i+1}:\n\nReasoning: Let's think step by step to {candidate_answers.completions.rationale[i]}\n\nAnswer: {candidate_answers.completions.answer[i]}\n\n" for i in range(len(candidate_answers.completions.answer))])
    
#     def forward(self, question):
#         candidate_answers = self.gen_candidate_answer(question=question)
#         create_candidate_answer_string = self.get_candidate_answers_string(candidate_answers=candidate_answers)
#         final_answer = self.gen_final_answer(candidate_answers=create_candidate_answer_string).final_answer
#         return dspy.Prediction(answer=final_answer)

# program = GSM8Reflection()
print(f"Evaluating baseline program...")
evaluate(program, devset=devset[:300])
breakpoint()
# Apply 0-shot MIPRO
from dspy.teleprompt import MIPROv2

# teleprompter = MIPROv2(
#     metric=gsm8k_metric,
#     num_candidates=10,
#     max_errors = 10000,
#     init_temperature=0.5,
#     verbose=False,
#     num_threads=4,
#     # log_dir="/lfs/0/kristaoo/dspy/examples/functional/logs",
# )

# zeroshot_optimized_program = teleprompter.compile(
#     program.deepcopy(),
#     trainset=trainset[:200],
#     max_bootstrapped_demos=0, # 0-shot optimization
#     max_labeled_demos=0,
#     num_batches=10,
#     # minibatch_size=25,
#     # minibatch_full_eval_steps=10, # TODO: should these be params in the compile step instead?
#     minibatch=False, 
#     seed=9,
#     requires_permission_to_run=False,
# )

# now = datetime.now()
# date_time = now.strftime("%Y%m%d_%H%M%S")
# zeroshot_optimized_program.save(f"zeroshot_mipro_optimized_{date_time}")

# print(f"Evluate optimized program...")
# evaluate(zeroshot_optimized_program, devset=devset[:300])


teleprompter = MIPROv2(
    metric=gsm8k_metric,
    num_candidates=10,
    max_errors = 10000,
    init_temperature=0.5,
    verbose=False,
    num_threads=4,
    # log_dir="/lfs/0/kristaoo/dspy/examples/functional/logs",
)

optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset[:200],
    max_bootstrapped_demos=3, # 0-shot optimization
    max_labeled_demos=4,
    num_trials=15,
    # minibatch_size=25,
    # minibatch_full_eval_steps=10, # TODO: should these be params in the compile step instead?
    minibatch=False, 
    seed=9,
    requires_permission_to_run=False,
)

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
optimized_program.save(f"mipro_optimized_{date_time}")

print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:300])