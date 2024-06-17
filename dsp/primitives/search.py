import logging
from collections.abc import Iterable

import numpy as np

import dsp

logger = logging.getLogger(__name__)

def retrieve(query: str, k: int, **kwargs) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=k, **kwargs)
    if not isinstance(passages, Iterable):
        # it's not an iterable yet; make it one.
        # TODO: we should unify the type signatures of dspy.Retriever
        passages = [passages]
    passages = [psg.long_text for psg in passages]

    if dsp.settings.reranker:
        passages_cs_scores = dsp.settings.reranker(query, passages)
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
        passages = [passages[idx] for idx in passages_cs_scores_sorted]


    return passages
def retrievewithMetadata(query: str, k: int, **kwargs) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""

    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=k, **kwargs)
    if not isinstance(passages, Iterable):
        # it's not an iterable yet; make it one.
        # TODO: we should unify the type signatures of dspy.Retriever
        passages = [passages]

    return passages


def retrieveRerankEnsemble(queries: list[str], k: int,**kwargs) -> list[str]:
    if not (dsp.settings.rm and dsp.settings.reranker):
        raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
    queries = [q for q in queries if q]
    passages = {}
    for query in queries:
        retrieved_passages = dsp.settings.rm(query, k=k*3,**kwargs)
        passages_cs_scores = dsp.settings.reranker(query, [psg.long_text for psg in retrieved_passages])
        for idx in np.argsort(passages_cs_scores)[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = passages.get(psg.long_text, []) + [
                passages_cs_scores[idx],
            ]


    passages = [(np.average(score), text) for text, score in passages.items()]
    return [text for _, text in sorted(passages, reverse=True)[:k]]

def retrieveRerankEnsemblewithMetadata(queries: list[str], k: int, **kwargs) -> list[str]:
    if not (dsp.settings.rm and dsp.settings.reranker):
        raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
    queries = [q for q in queries if q]
    all_queries_passages = []
    for query in queries:
        passages = []
        retrieved_passages = dsp.settings.rm(query, k=k * 3, **kwargs)
        passages_cs_scores = dsp.settings.reranker(
            query, passages=[psg["long_text"] for psg in retrieved_passages],
        )
        for idx in np.argsort(passages_cs_scores)[::-1][:k]:
            curr_passage = retrieved_passages[idx]
            curr_passage["rerank_score"] = passages_cs_scores[idx]
            passages.append(curr_passage)
        all_queries_passages.append(passages)
    if len(queries) == 1:
        return all_queries_passages[0]
    else:
        return all_queries_passages


def retrieveEnsemble(queries: list[str], k: int, by_prob: bool = True,**kwargs) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if dsp.settings.reranker:
        return retrieveRerankEnsemble(queries, k, **kwargs)
    queries = [q for q in queries if q]

    if len(queries) == 1:
        return retrieve(queries[0], k, **kwargs)

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, k=k * 3,**kwargs):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]


    return passages

def retrieveEnsemblewithMetadata(
    queries: list[str], k: int, by_prob: bool = True, **kwargs,
) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """

    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if not dsp.settings.reranker:
        return retrieveRerankEnsemblewithMetadata(queries=queries,k=k)

    queries = [q for q in queries if q]

    if len(queries) == 1:
        return retrieve(queries[0], k)
    all_queries_passages = []
    for q in queries:
        passages = {}
        retrieved_passages = dsp.settings.rm(q, k=k * 3, **kwargs)
        for idx, psg in enumerate(retrieved_passages):
            if by_prob:
                passages[(idx, psg.long_text)] = (
                    passages.get(psg.long_text, 0.0) + psg.prob
                )
            else:
                passages[(idx, psg.long_text)] = (
                    passages.get(psg.long_text, 0.0) + psg.score
                )
            retrieved_passages[idx]["tracking_idx"] = idx
        passages = sorted(passages.items(), key=lambda item: item[1])[:k]
        req_indices = [psg[0][0] for psg in passages]
        passages = [
            rp for rp in retrieved_passages if rp.get("tracking_idx") in req_indices
        ]
        all_queries_passages.append(passages)
    return all_queries_passages