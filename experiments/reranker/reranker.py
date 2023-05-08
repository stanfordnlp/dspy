import torch
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


import torch
from transformers import T5EncoderModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class T5EncoderRerank(torch.nn.Module):
    def __init__(self, model_type_or_dir):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(
            model_type_or_dir, torch_dtype="auto"
        )
        self.config = self.model.config
        self.first_transform = torch.nn.Linear(self.config.d_model, self.config.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.config.d_model, eps=1e-12)
        self.linear = torch.nn.Linear(self.config.d_model, 1)

    def forward(self, **kwargs):
        result = self.model(**kwargs).last_hidden_state[:, 0, :]
        first_transformed = self.first_transform(result)
        layer_normed = self.layer_norm(first_transformed)
        logits = self.linear(layer_normed)
        return SequenceClassifierOutput(logits=logits)


class T5EncoderRerankWrapper:
    def __init__(
        self,
        model_type_or_dir="t5-3b",
        path_checkpoint="trecdl22-crossencoder-rankT53b-repro/pytorch_model.bin",
    ):
        super().__init__()
        self.model = T5EncoderRerank(model_type_or_dir)
        self.model.load_state_dict(
            torch.load(path_checkpoint, map_location=torch.device("cpu"))
        )
        self.device = (
            torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)

    def __call__(self, query, passages, batch_size: int = 4):
        agg_scores = []
        for start_idx in range(0, len(passages), batch_size):
            end_idx = start_idx + batch_size

            to_tokenize = [[query, p] for p in passages[start_idx:end_idx]]
            features = self.tokenizer(
                to_tokenize, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                scores = self.model(**features).logits
            scores = [i[0] for i in scores.tolist()]
            agg_scores.extend(scores)

        assert len(agg_scores) == len(passages)
        return agg_scores


def retrieveRerankEnsemble(queries: list[str], k: int, reranker) -> list[str]:
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


# def retrieveRerankEnsembleAvg(
#     queries: list[str], k: int, reranker: SentenceTransformersCrossEncoder
# ) -> list[str]:
#     queries = [q for q in queries if q]

#     passages = {}
#     passages_org = {}
#     for q in queries:
#         retrieved_passages = dsp.settings.rm(q, k=100)
#         passages_cs_scores = reranker(q, [psg.long_text for psg in retrieved_passages])
#         passages_cs_scores_sorted = np.argsort(passages_cs_scores)
#         for idx in passages_cs_scores_sorted[::-1]:
#             psg = retrieved_passages[idx]
#             passages[psg.long_text] = passages.get(psg.long_text, []) + [
#                 passages_cs_scores[idx]
#             ]
#             passages_org[psg.long_text] = passages_org.get(psg.long_text, 0) + psg.prob

#     passages = [(np.average(score), text) for text, score in passages.items()]
#     passages = sorted(passages, reverse=True)  # [:k]
#     passages = [text for _, text in passages]

#     passages_org_scores = sorted(
#         [(score, text) for text, score in passages_org.items()], reverse=True
#     )
#     org_passages = [text for _, text in passages_org_scores]

#     return org_passages, passages


# def retrieveRerank(query: str, k: int, reranker) -> tuple[list[str], list[str]]:
#     org_passages = dsp.settings.rm(query, k=100)
#     passages_cs_scores = reranker(query, [psg.long_text for psg in org_passages])

#     passages_cs_scores = [np.mean(passages_cs_scores[idx], psg.prob) for idx, psg in enumerate(org_passages)]

#     passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
#     passages = [org_passages[idx].long_text for idx in passages_cs_scores_sorted]

#     passages_org_scores = {}
#     for psg in org_passages:
#         passages_org_scores[psg.long_text] = (
#             passages_org_scores.get(psg.long_text, 0) + psg.prob
#         )

#     passages_org_scores = sorted([(score, text) for text, score in passages_org_scores.items()], reverse=True)
#     org_passages = [text for _, text in passages_org_scores]

#     return org_passages, passages


def retrieveRerank(query: str, k: int, reranker) -> list[str]:
    org_passages = dsp.settings.rm(query, k=100)
    passages_cs_scores = reranker(query, [psg.long_text for psg in org_passages])

    passages_cs_scores = [
        passages_cs_scores[idx]*0.75 + psg.prob*0.25
        for idx, psg in enumerate(org_passages)
    ]

    passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
    passages = [org_passages[idx].long_text for idx in passages_cs_scores_sorted]

    return passages
def retrieveRerankEnsembleAvg(
    queries: list[str], k: int, reranker
) -> list[str]:
    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        retrieved_passages = dsp.settings.rm(q, k=100)
        passages_cs_scores = reranker(q, [psg.long_text for psg in retrieved_passages])
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)
        for idx in passages_cs_scores_sorted[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = passages.get(psg.long_text, []) + [
                passages_cs_scores[idx]
            ]

    passages = [(np.average(score), text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:100]
    passages = [text for _, text in passages]


    return passages