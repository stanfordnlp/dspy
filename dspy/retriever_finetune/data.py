import dspy
import random
from tqdm import tqdm
from dspy.datasets.dataset import Dataset
from dspy.retriever_finetune import GenerateSearchQuery, TEMPERATURE
import json
    
    
class RetrievalFinetuneDataset(Dataset):
    def __init__(self, *args, data: Dataset, passages_per_hop=1, max_hops=2, K=20, samples_per_query=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery, temperature=TEMPERATURE) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=K)
        self.max_hops = max_hops
        self.passages_per_hop = passages_per_hop
        self.samples_per_query = samples_per_query
        self.query_retrieval_history = {} # query -> topk psgs retrieved
        self.id_count = 0
        self.generate_dataset()
        
    def retrieve_topK(self, query):
        if query in self.query_retrieval_history:
            return self.query_retrieval_history[query]

        topK = self.retrieve(query).passages
        self.query_retrieval_history[query] = topK
        return topK
    
    def generate_dataset(self):
        dataset = []
        for row in tqdm(self.data):
            for _ in range(self.samples_per_query):
                context = []
                question = row['question']
                queries = []
                for hop in range(self.max_hops):
                    query = self.generate_query[hop](context=context, question=question).query
                    queries.append(query)
                    topK = self.retrieve_topK(query)
                    passages = random.sample(topK, self.passages_per_hop) # random sample from topK for now, later will use some probabilistic sampling
                    context = context + passages # deduplicate(context + passages)
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
            
