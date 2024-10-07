import random

import tqdm
from datasets import load_dataset

import dspy
from dspy.datasets.gsm8k import gsm8k_metric

from .base_task import BaseTask


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


class GSM8KTask(BaseTask):
    def __init__(self):
        dataset = load_dataset("gsm8k", "main")

        hf_official_train = dataset["train"]
        hf_official_test = dataset["test"]
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example["question"]

            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            official_train.append(
                dict(question=question, gold_reasoning=gold_reasoning, answer=answer)
            )

        for example in tqdm.tqdm(hf_official_test):
            question = example["question"]

            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            official_test.append(
                dict(question=question, gold_reasoning=gold_reasoning, answer=answer)
            )

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:2000]
        devset = official_train[2000:2500]
        testset = official_test[:]

        trainset = [dspy.Example(**x).with_inputs("question") for x in trainset]
        devset = [dspy.Example(**x).with_inputs("question") for x in devset]
        testset = [dspy.Example(**x).with_inputs("question") for x in testset]

        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.metric = gsm8k_metric

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return CoT()

    def get_metric(self):
        return self.metric

    def get_max_tokens(self):
        return 600
