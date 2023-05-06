import dsp
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder


class SentenceTransformersCrossEncoder:
    def __init__(
        self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        self.model = CrossEncoder(model_name_or_path)

    def __call__(self, query: str, passage: list[str]):
        return self.model.predict([(query, p) for p in passage])


def retrieveRerankEnsemble(
    queries: list[str], k: int, reranker: SentenceTransformersCrossEncoder
) -> list[str]:
    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        retrieved_passages = dsp.settings.rm(q, k=100)
        passages_cs_scores = reranker(q, [psg.long_text for psg in retrieved_passages])
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)
        for idx in passages_cs_scores_sorted[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = (
                passages.get(psg.long_text, 0.0) + passages_cs_scores[idx]
            )

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages

def retrieveRerankEnsembleAvg(
    queries: list[str], k: int, reranker: SentenceTransformersCrossEncoder
) -> list[str]:
    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        retrieved_passages = dsp.settings.rm(q, k=100)
        passages_cs_scores = reranker(q, [psg.long_text for psg in retrieved_passages])
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)
        for idx in passages_cs_scores_sorted[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = (
                passages.get(psg.long_text, []) + [passages_cs_scores[idx]]
            )

    passages = [(np.average(score), text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages

def retrieveRerank(query: str, k: int, reranker: SentenceTransformersCrossEncoder) -> list[str]:
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=100)
    passages_cs_scores = reranker(query, [psg.long_text for psg in passages])
    passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1][:k]
    passages = [passages[idx].long_text for idx in passages_cs_scores_sorted]

    return passages