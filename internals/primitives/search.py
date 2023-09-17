import numpy as np
import internals


def retrieve(query: str, k: int, **kwargs) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not internals.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = internals.settings.rm(query, k=k, **kwargs)
    passages = [psg.long_text for psg in passages]
    
    if internals.settings.reranker:
        passages_cs_scores = internals.settings.reranker(query, passages)
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
        passages = [passages[idx] for idx in passages_cs_scores_sorted]

    return passages


def retrieveRerankEnsemble(queries: list[str], k: int) -> list[str]:
    if not (internals.settings.rm and internals.settings.reranker):
        raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
    queries = [q for q in queries if q]
    passages = {}
    for query in queries:
        retrieved_passages = internals.settings.rm(query, k=k*3)
        passages_cs_scores = internals.settings.reranker(query, [psg.long_text for psg in retrieved_passages])
        for idx in np.argsort(passages_cs_scores)[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = passages.get(psg.long_text, []) + [
                passages_cs_scores[idx]
            ]

    passages = [(np.average(score), text) for text, score in passages.items()]
    return [text for _, text in sorted(passages, reverse=True)[:k]]


def retrieveEnsemble(queries: list[str], k: int, by_prob: bool = True) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not internals.settings.rm:
        raise AssertionError("No RM is loaded.")
    if internals.settings.reranker:
        return retrieveRerankEnsemble(queries, k)
    
    queries = [q for q in queries if q]

    if len(queries) == 1:
        return retrieve(queries[0], k)

    passages = {}
    for q in queries:
        for psg in internals.settings.rm(q, k=k * 3):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages
