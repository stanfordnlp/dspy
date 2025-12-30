import dspy
import dspy.evaluate
from dspy.datasets import HotPotQA
from dspy.dsp.utils import deduplicate
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.bootstrap import BootstrapFewShot


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)


def load_hotpotqa():
    # Load the dataset.
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]
    return trainset, devset


# @pytest.mark.slow_test
# TODO: Find a way to make this test run without openai
def _test_baleen():
    lm = dspy.OpenAI(model="gpt-3.5-turbo")
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    dspy.configure(lm=lm, rm=rm)

    # Ask any question you like to this simple RAG program.
    my_question = "How many storeys are in the castle that David Gregory inherited?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
    pred = uncompiled_baleen(my_question)

    assert pred.answer == "five"


def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False
    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if "query" in outputs]

    if max([len(h) for h in hops]) > 100:
        return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))):
        return False

    return True


def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context]))

    return gold_titles.issubset(found_titles)


# @pytest.mark.slow_test
# TODO: Find a way to make this test run without the slow hotpotqa dataset
def _test_compiled_baleen():
    trainset, devset = load_hotpotqa()
    lm = dspy.OpenAI(model="gpt-3.5-turbo")
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    dspy.configure(lm=lm, rm=rm)

    uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program

    teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
    compiled_baleen = teleprompter.compile(
        SimplifiedBaleen(),
        teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=trainset,
    )

    evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    uncompiled_baleen_retrieval_score = evaluate_on_hotpotqa(
        uncompiled_baleen, metric=gold_passages_retrieved, display=False
    )
    # assert uncompiled_baleen_retrieval_score / 100 == 18 / 50

    compiled_baleen_retrieval_score = evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved)
    # assert compiled_baleen_retrieval_score / 100 == 27 / 50
    assert uncompiled_baleen_retrieval_score < compiled_baleen_retrieval_score
