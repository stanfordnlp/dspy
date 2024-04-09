from collections.abc import Iterable
import warnings
import numpy as np

import dsp


def retrieve(query: str, k: int, **kwargs) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if not dsp.settings.reranker:
        warnings.warn("If you want to use the Reranker, please use dspy.RetrieveThenRerank")
    passages = dsp.settings.rm(query, k=k, **kwargs)
    if not isinstance(passages, Iterable):
        # it's not an iterable yet; make it one.
        # TODO: we should unify the type signatures of dspy.Retriever
        passages = [passages]
    # passages = [psg.long_text for psg in passages]
    
    # if dsp.settings.reranker:
    #     passages_tracking_idx = {str(idx):psg for idx, psg in enumerate(passages)}
    #     passages_long_text = [psg.long_text for psg in passages]
    #     passages_cs_scores = dsp.settings.reranker(query, passages_long_text)
    #     passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
    #     passages = [passages_long_text[idx] for idx in passages_cs_scores_sorted]

    return passages


def retrieveRerankEnsemble(queries: list[str], k: int,**kwargs) -> list[str]:
    if not (dsp.settings.rm and dsp.settings.reranker):
        raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
    queries = [q for q in queries if q]
    all_queries_passages = []
    for query in queries:
        passages = []
        retrieved_passages = dsp.settings.rm(query, k=k*3,**kwargs)
        passages_cs_scores = dsp.settings.reranker(query,passages=[psg["long_text"] for psg in retrieved_passages])
        for idx in np.argsort(passages_cs_scores)[::-1][:k]:
            curr_passage = retrieved_passages[idx]
            curr_passage['rerank_score'] = passages_cs_scores[idx]
            passages.append(curr_passage)
        all_queries_passages.append(passages)
    if len(queries) == 1:
        return all_queries_passages[0]
    else:
        return all_queries_passages

# def retrieveRerankEnsemble(queries: list[str], k: int,**kwargs) -> list[str]:
#     if not (dsp.settings.rm and dsp.settings.reranker):
#         raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
#     queries = [q for q in queries if q]
#     passages = {}
#     for query in queries:
#         retrieved_passages = dsp.settings.rm(query, k=k*3,**kwargs)
#         passages_cs_scores = dsp.settings.reranker(query, [psg.long_text for psg in retrieved_passages])
#         for idx in np.argsort(passages_cs_scores)[::-1]:
#             psg = retrieved_passages[idx]
#             passages[psg.long_text] = passages.get(psg.long_text, []) + [
#                 passages_cs_scores[idx],
#             ]

#     passages = [(np.average(score), text) for text, score in passages.items()]
#     return [text for _, text in sorted(passages, reverse=True)[:k]]

def retrieveEnsemble(queries: list[str], k: int, by_prob: bool = True,**kwargs) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if not dsp.settings.reranker:
        warnings.warn("If you want to use the Reranker, please use dspy.RetrieveThenRerank. The reranking is ignored here.")
    
    queries = [q for q in queries if q]

    if len(queries) == 1:
        return retrieve(queries[0], k)
    all_queries_passages = []
    for q in queries:
        passages = {}
        retrieved_passages =  dsp.settings.rm(q, k=k * 3,**kwargs)
        # for idx,psg in enumerate(retrieved_passages):
        #     retrieved_passages[idx]["tracking_idx"] = idx
        for idx,psg in enumerate(retrieved_passages):
            if by_prob:
                passages[(idx,psg.long_text)] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[(idx,psg.long_text)] = passages.get(psg.long_text, 0.0) + psg.score
            retrieved_passages[idx]["tracking_idx"] = idx
        # passages = [(score, text) for text, score in passages.items()]
        passages = sorted(passages.items(), key=lambda item: item[1])[:k]
        # passages = sorted(passages, reverse=True)[:k]
        req_indices = [psg[0][0] for psg in passages]
        passages = [rp for rp in retrieved_passages if rp.get("tracking_idx") in req_indices]
        all_queries_passages.append(passages)
    return all_queries_passages
