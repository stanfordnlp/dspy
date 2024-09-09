import json
import dspy
from dspy.datasets import HotPotQA

combined_res = []
num_errors = 0
for i in range(3):
    res = json.load(open(f"results_8b_{i*500}_{(i+1)*500}.json", "r"))
    num_errors += sum(1 for r in res if r.get("error", None) is not None)
    res = [r for r in res if r.get("error", None) is None]
    combined_res.extend(res)
print(len(combined_res))
print(num_errors)

dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
QA_map = {x["question"]: x for x in devset}

predictions = [dspy.Prediction(answer=x["answer"], question=x["question"]) for x in combined_res]
metric = dspy.evaluate.answer_exact_match
results = [metric(x, QA_map[x.question]) for x in predictions]
print("results: ", sum(results), "/", len(results), f"= {sum(results) / len(results)}")
