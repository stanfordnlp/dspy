import dspy
from dsp.utils import deduplicate
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot

# pipeline configs
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)


# load dataset
dataset = HotPotQA(
    train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0
)
trainset = [x.with_inputs("question") for x in dataset.train]
devset = [x.with_inputs("question") for x in dataset.dev]


# signatures of dspy modules
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


failure_counts = {
    "answer_exact_match": 0,
    "answer_passage_match": 0,
    "max_hop_length_exceeded": 0,
    "hop_query_similarity_exceeded": 0,
    "failed_prog_assertions": 0,
}


def validate_context_and_answer_and_hops(example, pred, trace=None):
    global failure_counts

    try:
        if not dspy.evaluate.answer_exact_match(example, pred):
            failure_counts["answer_exact_match"] += 1
            return False

        if not dspy.evaluate.answer_passage_match(example, pred):
            failure_counts["answer_passage_match"] += 1
            return False

        hops = [example.question] + [
            outputs.query for *_, outputs in trace if "query" in outputs
        ]

        if max([len(h) for h in hops]) > 100:
            failure_counts["max_hop_length_exceeded"] += 1
            return False

        if any(
            dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
            for idx in range(2, len(hops))
        ):
            failure_counts["hop_query_similarity_exceeded"] += 1
            return False
    except:
        failure_counts["failed_prog_assertions"] += 1
        print("failed prog assertions:", example.question)
        return False

    return True


def validate_query_distinction_local(previous_queries, query):
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True


# declaration of dspy program
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query

            dspy.Assert(
                len(query) <= 100,
                "Query should be short and less than 100 characters",
            )

            dspy.Assert(
                validate_query_distinction_local(prev_queries, query),
                "Query should not be the following: "
                + "; ".join(
                    f"{idx+1}) {query}" for idx, query in enumerate(prev_queries)
                ),
            )

            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)

        return dspy.Prediction(context=context, answer=pred.answer)


teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
compiled_baleen = teleprompter.compile(
    SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset
)

print("=" * 50, "Validation Failures", "=" * 50)
for failure_type, count in failure_counts.items():
    print(f"{failure_type}: {count}")
