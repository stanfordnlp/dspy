from collections.abc import Callable
from typing import Any, Optional

import dsp
from dsp.utils import deduplicate

Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
Answer = dsp.Type(
    prefix="Answer:",
    desc="${a short factoid answer, often between 1 and 5 words}",
    format=dsp.format_answers,
)

Context = dsp.Type(
    prefix="Context:\n",
    desc="${sources that may contain relevant content}",
    format=dsp.passages2text,
)

SearchRationale = dsp.Type(
    prefix="Rationale: Let's think step by step. To answer this question, we first need to find out",
    desc="${the missing information}",
)

SearchQuery = dsp.Type(
    prefix="Search Query:",
    desc="${a simple question for seeking the missing information}",
)
Rationale = dsp.Type(
    prefix="Rationale: Let's think step by step.",
    desc="${a step-by-step deduction that identifies the correct response, which will be provided below}",
)

# basic QA predict for demonstration
qa_template_with_CoT = dsp.Template(
    instructions="Answer questions with short factoid answers.",
    context=Context(),
    question=Question(),
    rationale=Rationale(),
    answer=Answer(),
)

# for hop-0
rewrite_template = dsp.Template(
    instructions="Write a search query that will help answer a complex question.",
    question=Question(),
    rationale=SearchRationale(),
    query=SearchQuery(),
)

CondenseRationale = dsp.Type(
    prefix="Rationale: Let's think step by step. Based on the context, we have learned the following.",
    desc="${information from the context that provides useful clues}",
)

# hop-1+
hop_template = dsp.Template(
    instructions=rewrite_template.instructions,
    context=Context(),
    question=Question(),
    rationale=CondenseRationale(),
    query=SearchQuery(),
)

# ------------------------- Start of Demonstrate -------------------------


@dsp.transformation
def multihop_search_v1(example: dsp.Example, max_hops=2, k=2) -> dsp.Example:
    example.context = []

    for hop in range(max_hops):
        # Generate a query based
        template = rewrite_template if hop == 0 else hop_template
        example, completions = dsp.generate(template)(example, stage=f"h{hop}")

        # Retrieve k results based on the query generated
        passages = dsp.retrieve(completions.query, k=k)

        # Update the context by concatenating old and new passages
        example.context = deduplicate(example.context + passages)

    return example


@dsp.transformation
def QA_predict(example: dsp.Example, sc=False):
    if sc:
        raise NotImplementedError # just to make sure we aren't doing this
    else:
        example, completions = dsp.generate(qa_template_with_CoT)(example, stage="qa")

    return example.copy(answer=completions.answer)


@dsp.transformation
def multihop_attempt(d: dsp.Example, demos: list[dsp.Example]) -> dsp.Example:
    # Prepare unaugmented demonstrations for the example.
    x = dsp.Example(question=d.question, demos=dsp.all_but(demos, d))

    # Search. And skip examples where search fails.
    # Annotate demonstrations for multihop_search_v2 with the simpler multihop_search_v1 pipeline.
    x = multihop_search_v1(x)
    if not dsp.passage_match(x.context, d.answer):
        return None

    # Predict. And skip examples where predict fails.
    x = QA_predict(x, sc=False)
    if not dsp.answer_match(x.answer, d.answer):
        return None

    return d.copy(**x)


@dsp.transformation
def multihop_demonstrate(x: dsp.Example, train: list[dsp.Example]) -> dsp.Example:
    """No SC for now."""
    demos = dsp.sample(train, k=7)
    x.demos = dsp.annotate(multihop_attempt)(demos, k=3, return_all=True, demos=train)
    return x


# ------------------------- END of Demonstrate ---------------------------
# ------------------------- Start of predict -----------------------------


@dsp.transformation
def multihop_search_v2(
    example: dsp.Example,
    ranking_fnc: Callable = dsp.retrieveEnsemble,
    max_hops=2,
    k=5,
    ranking_params: Optional[dict[str, Any]] = None,
) -> dsp.Example:
    example.context = []

    for hop in range(max_hops):
        # Generate queries
        template = rewrite_template if hop == 0 else hop_template
        # reducing n to 1 and temperature to 0.1 for simplicity.
        example, completions = dsp.generate(template, n=1, temperature=0.1)(
            example, stage=f"h{hop}"
        )

        # Collect the queries and search with result fusion
        queries = [c.query for c in completions] + [example.question]

        # where re-ranking changes
        # example.context = dsp.retrieveEnsemble(queries, k=k)
        ranking_params = ranking_params or {}
        if k not in ranking_params:
            ranking_params["k"] = k
        example.context = ranking_fnc(queries, **ranking_params)

        # Arrange the passages for the next hop
        if hop > 0:
            example.context = [completions[0].rationale] + example.context

    return example


# ------------------------- END of predict -------------------------


def multihop_QA(
    question: str,
    train: list[dsp.Example],
    ranking_fnc: Callable = dsp.retrieveEnsemble,
    ranking_params: Optional[dict[str, Any]] = None,
) -> str:
    x = dsp.Example(question=question)
    # no optimization on demonstration phase
    x = multihop_demonstrate(x, train)
    x = multihop_search_v2(x, ranking_fnc=ranking_fnc, ranking_params=ranking_params)
    x = QA_predict(x)
    return x.answer
