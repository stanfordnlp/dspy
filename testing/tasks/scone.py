import glob
import os
import random

import pandas as pd

import dspy

from .base_task import BaseTask


def load_scone(dirname):
    dfs = []
    for filename in glob.glob(dirname + "/*.csv"):
        df = pd.read_csv(filename, index_col=0)
        df["category"] = os.path.basename(filename).replace(".csv", "")
        dfs.append(df)
    data_df = pd.concat(dfs)

    def as_example(row):
        # The 'one_scoped' file is from an earlier dataset, MoNLI, and
        # so is formatted a bit differently:
        suffix = "" if row["category"] == "one_scoped" else "_edited"
        # Reformat the hypothesis to be an embedded clause in a question:
        hkey = "sentence2" + suffix
        question = row[hkey][0].lower() + row[hkey][1:].strip(".")
        question = f"Can we logically conclude for sure that {question}?"
        # Binary task formulation:
        label = "Yes" if row["gold_label" + suffix] == "entailment" else "No"
        return dspy.Example(
            {
                "context": row["sentence1" + suffix],
                "question": question,
                "answer": label,
                "category": row["category"],
            }
        ).with_inputs("context", "question")

    return list(data_df.apply(as_example, axis=1).values)


class ScoNeSignature(dspy.Signature):
    ("""context, question -> answer""")

    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Yes or No")


class ScoNeCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(ScoNeSignature)

    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)


class ScoNeTask(BaseTask):
    def __init__(self):
        # !git clone https://github.com/selenashe/ScoNe.git
        if not os.path.exists("ScoNe"):
            os.system("git clone https://github.com/selenashe/ScoNe.git")

        # Load and configure the datasets.
        all_train = load_scone("ScoNe/scone_nli/train")

        random.seed(1)
        random.shuffle(all_train)

        # 1000 random train, 500 random dev:
        self.trainset, self.testset = all_train[:1000], all_train[1000:1500]

        metric_EM = dspy.evaluate.answer_exact_match
        self.metric = metric_EM

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return ScoNeCoT()

    def get_metric(self):
        return self.metric

    def get_max_tokens(self):
        return 200
