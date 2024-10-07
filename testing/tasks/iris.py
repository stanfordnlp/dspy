from datasets import load_dataset
from .base_task import BaseTask

import random
import dspy


class Sig(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."

    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc="setosa, versicolor, or virginica")


class Classify(dspy.Module):
    def __init__(self):
        self.pred = dspy.ChainOfThought(Sig)

    def forward(self, petal_length, petal_width, sepal_length, sepal_width):
        return self.pred(
            petal_length=petal_length,
            petal_width=petal_width,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
        )


class IrisClassifierTask(BaseTask):
    def __init__(self):
        # Read in the conditional HotpotQA dataset from nfl_datasets as a csv from nfl_datasets/conditional_hotpotqa
        dataset = load_dataset("hitorilabs/iris")

        fullset = [
            dspy.Example(**{k: str(round(v, 2)) for k, v in example.items()})
            for example in dataset["train"]
        ]
        fullset = [
            dspy.Example(
                **{
                    **x,
                    "answer": ["setosa", "versicolor", "virginica"][int(x["species"])],
                }
            )
            for x in fullset
        ]
        fullset = [
            x.with_inputs("petal_length", "petal_width", "sepal_length", "sepal_width")
            for x in fullset
        ]

        random.Random(0).shuffle(fullset)
        # self.trainset, self.devset, self.testset = fullset[:25], fullset[20:75], fullset[75:]
        self.trainset, self.testset = fullset[:75], fullset[75:]

        # Set up metrics
        NUM_THREADS = 16

        # TODO: set up metrics
        self.metric = dspy.evaluate.answer_exact_match

        kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=15)

        self.set_splits(TRAIN_NUM=25, DEV_NUM=55, TEST_NUM=100)

    def get_program(self):
        return Classify()

    def get_metric(self):
        return self.metric

    def get_default_max_bootstrapped_demos(self):
        return 4

    def get_default_max_labeled_demos(self):
        return 0

    def get_handwritten_prompts(self):
        return []

    def get_trainset(self, TRAIN_NUM=None):
        return self.trainset

    def get_testset(self):
        return self.testset

    def get_devset(self, TRAIN_NUM=None, DEV_NUM=None):
        return self.trainset

    def get_evalset(self, EVAL_NUM=None):
        return self.testset
