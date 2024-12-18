import random

import tqdm
from datasets import load_dataset

from dspy.evaluate import Evaluate, normalize_text
import dspy

from .base_task import BaseTask


def count_unique_docs(example):
    return len(set(fact["key"] for fact in example["supporting_facts"]))


class RetrieveMultiHop2(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=16)
        self.create_query = dspy.ChainOfThought("claim,summary->query")
        self.summarize = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        passages = self.retrieve(claim).passages
        summary = ""

        for _ in range(2):
            summary = self.summarize(claim=claim, context=summary, passages=passages).summary
            query = self.create_query(claim=claim, summary=summary).query
            passages.extend(self.retrieve(query).passages)

        return dspy.Prediction(retrieved_docs=passages)


class RetrieveMultiHop4(dspy.Module):
    def __init__(self):
        super().__init__()
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=7)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

        # HOP 3
        hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverRetrieveDiscrete(BaseTask):
    def __init__(self):
        dataset = load_dataset("hover")

        hf_trainset = dataset["train"]
        hf_testset = dataset["validation"]

        reformatted_hf_trainset = []
        reformatted_hf_testset = []

        for example in tqdm.tqdm(hf_trainset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            if count_unique_docs(example) == 3:  # Limit to 3 hop examples
                reformatted_hf_trainset.append(dict(claim=claim, supporting_facts=supporting_facts, label=label))

        for example in tqdm.tqdm(hf_testset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            if count_unique_docs(example) == 3:
                reformatted_hf_testset.append(dict(claim=claim, supporting_facts=supporting_facts, label=label))

        rng = random.Random(1)
        rng.shuffle(reformatted_hf_trainset)
        rng = random.Random(2)
        rng.shuffle(reformatted_hf_testset)

        trainset = reformatted_hf_trainset
        testset = reformatted_hf_testset

        self.trainset = [dspy.Example(**x).with_inputs("claim") for x in trainset]
        self.testset = [dspy.Example(**x).with_inputs("claim") for x in testset]

        # Set up metrics
        NUM_THREADS = 16

        kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=15)
        self.evaluate = Evaluate(devset=self.trainset, metric=self.get_metric(), **kwargs)

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return RetrieveMultiHop4()

    def get_metric(self):
        def discrete_retrieval_eval(example, pred, trace=None):
            gold_titles = [fact["key"] for fact in example["supporting_facts"]]
            found_titles = [c.split(" | ")[0] for c in pred.retrieved_docs]
            return set(map(normalize_text, gold_titles)) <= set(map(normalize_text, found_titles))

        return discrete_retrieval_eval

    def get_default_max_bootstrapped_demos(self):
        return 1

    def get_default_max_labeled_demos(self):
        return 2

    def get_max_tokens(self):
        return 600
