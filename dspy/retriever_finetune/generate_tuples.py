import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.datasets import HotPotQA
from dspy.evaluate.evaluate import Evaluate
from dspy.retriever_finetune import RetrievalFinetuneDataset, RetrievalFinetuneGenerator, validate_context_and_answer_and_hops
from collections import defaultdict
import json



if __name__ == '__main__':
    dataset = HotPotQA(train_seed=1, train_size=50, eval_seed=2023, dev_size=20, test_size=0)
    hotpot_trainset = [x.with_inputs('question') for x in dataset.train]
    
    colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    # llama2 = dspy.HFModel(model = 'meta-llama/Llama-2-7b-chat-hf')
    turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=128)
    dspy.settings.configure(rm=colbert, lm=turbo)
    
    multihop_dataset = RetrievalFinetuneDataset(data=hotpot_trainset, K=10, samples_per_query=3)
    multihop_dataset.save_train('data/dataset-train-colbert-gpt-3.5-turbo.json')
    multihop_dataset.save_dev('data/dataset-dev-colbert-gpt-3.5-turbo.json')
    
    trainset = [x.with_inputs('question', 'context') for x in multihop_dataset.train]
    evalset = [x.with_inputs('question', 'context') for x in multihop_dataset.dev]
    # uncompiled_generator = RetrievalFinetuneGenerator()

    teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
    compiled_generator = teleprompter.compile(RetrievalFinetuneGenerator(), teacher=RetrievalFinetuneGenerator(), trainset=trainset)
    
    evaluate_on_hotpotqa = Evaluate(devset=evalset, num_threads=32, display_progress=False, display_table=False)
    metric = dspy.evaluate.answer_exact_match
    scores = evaluate_on_hotpotqa(compiled_generator, metric=metric, return_all_scores=True, display_progress=False, display_table=5)
    print(scores)
    
    passage_map = defaultdict(dict)
    for row, score in zip(evalset, scores[1]):
        seen_tuples = set()
        for q, c in zip(row['queries'], row['context']): # for passages A -> B, A is positive if any B makes the downstream metric True
            seen_tuples.add((q, c))
            if score:
                for sq, sc in seen_tuples:
                    passage_map[sq][sc] = True
            else:
                passage_map[q][c] = False
    
    print(passage_map)
    with open('data/passage-map-gpt-3.5-turbo.json', 'w') as f:
        json.dump(passage_map, f)
        
    
    