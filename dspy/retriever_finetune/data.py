import dspy
import random
from tqdm import tqdm
from dspy.datasets.dataset import Dataset
from dspy.retriever_finetune import GenerateSearchQuery, TEMPERATURE
import json
import numpy as np
from collections import defaultdict
from itertools import combinations
import math
    
    
class RetrievalFinetuneDataset(Dataset):
    def __init__(self, *args, data: Dataset, passages_per_hop=1, max_hops=2, K=20, samples_per_query=10, perturb=True, **kwargs):
        super().__init__(*args, **kwargs)
        assert samples_per_query < math.comb(K, passages_per_hop)
        self.data = data
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery, temperature=TEMPERATURE) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=K)
        self.max_hops = max_hops
        self.passages_per_hop = passages_per_hop
        self.samples_per_query = samples_per_query
        self.k = K
        self.query_retrieval_history = {} # query -> topk psgs retrieved
        self.tuple_history = defaultdict(set) # query -> set of passage tuples we have already seen
        self.id_count = 0
        self.perturb = perturb
        if perturb:
            self.sample_weights = np.power(np.full(K, 0.8), np.arange(K))
            self.sample_weights = self.sample_weights / np.sum(self.sample_weights)
        else:
            self.combs = list(combinations(range(K), passages_per_hop))
            self.combs.sort(key=lambda x:sum(x))
        self.rng = np.random.default_rng(seed=0)
        self.generate_dataset()
        
    def retrieve_topK(self, query):
        if query in self.query_retrieval_history:
            return self.query_retrieval_history[query]

        topK = self.retrieve(query).passages
        self.query_retrieval_history[query] = topK
        return topK
    
    def retrieve_and_sample(self, query):
        # can be really slow if samples_per_query is just less than comb(K, passages_per_hop), could sample from all perms -> remove seen tuples, but that would be memory expensive
        topK = self.retrieve_topK(query)
        comb_index = 0
        while True:
            if self.perturb:
                passage_indices = tuple(self.rng.choice(np.arange(self.k), size=self.passages_per_hop,replace=False, p=self.sample_weights))
            else:
                passage_indices = self.combs[comb_index]
                comb_index += 1
            if passage_indices not in self.tuple_history[query]:
                self.tuple_history[query].add(passage_indices)
                return [topK[i] for i in passage_indices]
    
    def generate_dataset(self):
        dataset = []
        for row in tqdm(self.data, desc='Creating dataset'):
            for _ in range(self.samples_per_query):
                context = []
                question = row['question']
                queries = []
                for hop in range(self.max_hops):
                    query = self.generate_query[hop](context=context, question=question).query
                    queries.append(query)
                    passages = self.retrieve_and_sample(query)
                    context = context + passages # deduplicate(context + passages)
                    # context = [item[:500] for item in context]
                dataset.append({'id': self.id_count, 'question': question, 'context': context, 'answer': row['answer'], 'queries': queries}) # 'supporting_facts': row['supporting_facts']
                self.id_count += 1
                
        rng = random.Random(0)
        rng.shuffle(dataset)
        self._train = dataset[:len(dataset)*75//100]
        self._dev = dataset[len(dataset)*75//100:]
        
    def save_dev(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self._dev, f)
        
    def save_train(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self._train, f)
            
