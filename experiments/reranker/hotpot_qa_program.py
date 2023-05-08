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
def QA_predict(example: dsp.Example, n=20):
    if n == 0:
        return example.copy(answer="")

    if n > 1:
        example, completions = dsp.generate(qa_template_with_CoT, n=n, temperature=0.7)(
            example, stage="qa"
        )
        completions = dsp.majority(completions)
    else:
        example, completions = dsp.generate(qa_template_with_CoT)(example, stage="qa")

    return example.copy(answer=completions.answer)


@dsp.transformation
def multihop_search(
    example: dsp.Example,
    max_hops=2,
    num_queries=10,
    k=7,
    ranking_fnc_col: list[Callable] = [dsp.retrieveEnsemble, dsp.retrieveEnsemble],
    ranking_params_col: list[dict[str, Any]] = None,
) -> dsp.Example:
    example.background = []
    example.context = []

    ranking_params = ranking_params_col[0]
    ranking_params_new = ranking_params_col[1]

    ranking_params = ranking_params or {}
    if k not in ranking_params:
        ranking_params["k"] = k

    ranking_params_new = ranking_params_new or {}
    if k not in ranking_params_new:
        ranking_params_new["k"] = k

    ranking_fnc = ranking_fnc_col[0]
    ranking_fnc_new = ranking_fnc_col[1]
    for hop in range(max_hops):
        # Generate queries
        template = rewrite_template if hop == 0 else hop_template

        if num_queries == 1:
            example, completions = dsp.generate(template)(example, stage=f"h{hop}")
            # passages = dsp.retrieve(completions.query, k=k)
            org_passages = ranking_fnc(completions.query, **ranking_params)
            passages = ranking_fnc_new(completions.query, **ranking_params_new)

        else:
            raise NotImplementedError
            # num_queries = int(num_queries)
            # example, completions = dsp.generate(
            #     template, n=num_queries, temperature=0.7
            # )(example, stage=f"h{hop}")
            # queries = [c.query for c in completions] + [example.question]
            # # passages = dsp.retrieveEnsemble(queries, k=k)
            # passages = ranking_fnc(queries, **ranking_params)

        # Arrange the passages for the next hop
        if hop == 0:
            example.context = passages
            org_context = org_passages
        else:
            example.context = [completions[0].rationale] + example.background + passages
            org_context = [completions[0].rationale] + example.background + org_passages

        example[f"h{hop}_copy_reranked"] = dsp.dotdict(
            {"context": deduplicate(example.context)}
        )
        example[f"h{hop}_copy_org"] = dsp.dotdict({"context": deduplicate(org_context)})

        example.context = deduplicate(example.context)[:k]
        example.background = deduplicate(example.background + passages)[: hop + 1]

    return example

@dsp.transformation
def multihop_search_old(
    example: dsp.Example,
    max_hops=2,
    num_queries=10,
    k=7,
    ranking_fnc: Callable = dsp.retrieveEnsemble,
    ranking_params: dict[str, Any] = None,
) -> dsp.Example:
    example.background = []
    example.context = []


    ranking_params = ranking_params or {}
    if k not in ranking_params:
        ranking_params["k"] = k

    for hop in range(max_hops):
        # Generate queries
        template = rewrite_template if hop == 0 else hop_template

        if num_queries == 1:
            example, completions = dsp.generate(template)(example, stage=f"h{hop}")
            # passages = dsp.retrieve(completions.query, k=k)
            passages = ranking_fnc(completions.query, **ranking_params)

        else:
            raise NotImplementedError

        # Arrange the passages for the next hop
        if hop == 0:
            example.context = passages
        else:
            example.context = [completions[0].rationale] + example.background + passages

        example[f"h{hop}_copy_reranked"] = dsp.dotdict(
            {"context": deduplicate(example.context)}
        )

        example.context = deduplicate(example.context)[:k]
        example.background = deduplicate(example.background + passages)[: hop + 1]

    return example


@dsp.transformation
def multihop_attempt(d: dsp.Example, demos: list[dsp.Example]):
    x = dsp.Example(question=d.question, demos=dsp.all_but(demos, d))
    x = multihop_search_old(x, num_queries=1, k=4, ranking_fnc=dsp.retrieve)

    if not dsp.passage_match(x.context, d.answer):
        return None
    x = QA_predict(x, n=1)

    if not dsp.answer_match(x.answer, d.answer):
        return None
    return d.copy(**x)


@dsp.transformation
def multihop_demonstrate(x: dsp.Example, train: list[dsp.Example]):
    demos = dsp.sample(train, k=7)
    x.demos = dsp.annotate(multihop_attempt)(demos, k=3, return_all=True, demos=train)
    return x


# ------------------------- END of Demonstrate ---------------------------
# ------------------------- Start of predict -----------------------------


def multihop_QA(
    question: str,
    train: list[dsp.Example],
    num_queries=1,
    num_preds=0,
    config={"org_ranking_fnc": dsp.retrieve, "org_ranking_params": {"k": 7}},
    # ranking_fnc: Callable = dsp.retrieveEnsemble,
    # ranking_params: Optional[dict[str, Any]] = None,
):
    x = dsp.Example(question=question)
    x = multihop_demonstrate(x, train)

    ranking_fnc, ranking_params = (
        config["org_ranking_fnc"],
        config["org_ranking_params"],
    )
    new_ranking_fnc, new_ranking_params = (
        config["new_ranking_fnc"],
        config["new_ranking_params"],
    )
    y = x.copy()
    y = multihop_search(
        y,
        num_queries=num_queries,
        ranking_fnc_col=[ranking_fnc, new_ranking_fnc],
        ranking_params_col=[ranking_params, new_ranking_params],
    )
    y = QA_predict(y, n=num_preds)

    # if "new_ranking_fnc" not in config:
    #     return y
    # ranking_fnc, ranking_params = config["new_ranking_fnc"], config["new_ranking_params"]
    # z = x.copy()
    # z = multihop_search(z, num_queries=num_queries, ranking_fnc=ranking_fnc, ranking_params=ranking_params)
    # z = QA_predict(z, n=num_preds)

    return y


# ------------------------- END of predict -------------------------
