import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate

from .base_task import BaseTask


class MultiHop(dspy.Module):
    def __init__(self, passages_per_hop):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context ,question->search_query")
        self.generate_answer = dspy.ChainOfThought("context ,question->answer")

    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return dspy.Prediction(
            context=context,
            answer=self.generate_answer(context=context, question=question).answer,
        )


class HotPotQATask(BaseTask):
    def __init__(self):
        # Load and configure the datasets.
        hotpot_dataset = HotPotQA(train_seed=1, eval_seed=2023)

        self.trainset = [x.with_inputs("question") for x in hotpot_dataset.train]
        self.testset = [x.with_inputs("question") for x in hotpot_dataset.dev]

        # Set up metrics
        NUM_THREADS = 16

        metric_EM = dspy.evaluate.answer_exact_match
        self.metric = metric_EM

        def gold_passages_retrieved(example, pred, trace=None):
            gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
            found_titles = set(
                map(
                    dspy.evaluate.normalize_text,
                    [c.split(" | ")[0] for c in pred.context],
                )
            )
            return gold_titles.issubset(found_titles)

        kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=15)
        self.evaluate_EM = Evaluate(devset=self.trainset, metric=metric_EM, **kwargs)
        self.evaluate_retrieval = Evaluate(
            devset=self.trainset, metric=gold_passages_retrieved, **kwargs
        )

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return MultiHop(passages_per_hop=3)

    def get_metric(self):
        return self.metric

    def get_retrieval_metric(self):
        return self.evaluate_retrieval

    def get_default_max_bootstrapped_demos(self):
        return 1

    def get_default_max_labeled_demos(self):
        return 2
