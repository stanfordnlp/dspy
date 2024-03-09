import json
from typing import Union

from datasets import Dataset

from dsp.utils import dotdict


class PyseriniRetriever:
    """Wrapper for retrieval with Pyserini. Supports using either pyserini prebuilt faiss indexes or your own faiss index."""

    def __init__(self, 
                 query_encoder: str = 'castorini/dkrr-dpr-nq-retriever', 
                 index: str = 'wikipedia-dpr-dkrr-nq', 
                 dataset: Dataset = None,
                 id_field: str = '_id',
                 text_fields: list[str] = ['text']) -> None:
        """
        Args:
        
            query_encoder (`str`):
                Huggingface model to encode queries
            index (`str`):
                Either a prebuilt index from pyserini or a local path to a faiss index
            dataset (`Dataset`):
                Only required when using a local faiss index. The dataset should be the one that has been put into the faiss index.
            id_field (`str`):
                The name of the id field of the dataset used for retrieval.
            text_fields (`list[str]`):
                A list of the names of the text fields for the dataset used for retrieval.
        """
        
        # Keep pyserini as an optional dependency
        from pyserini.prebuilt_index_info import FAISS_INDEX_INFO, IMPACT_INDEX_INFO, TF_INDEX_INFO
        from pyserini.search import FaissSearcher
        
        self.encoder = FaissSearcher._init_encoder_from_str(query_encoder)
        self.dataset = dataset
        self.id_field = id_field
        self.text_fields = text_fields
        
        if index in TF_INDEX_INFO or index in FAISS_INDEX_INFO or index in IMPACT_INDEX_INFO:
            self.searcher = FaissSearcher.from_prebuilt_index(index, self.encoder)
        else:
            self.searcher = FaissSearcher(index_dir=index, query_encoder=self.encoder)
            assert self.dataset is not None
            self.dataset_id_to_index = {}
            for i, docid in enumerate(self.dataset[self.id_field]):
                self.dataset_id_to_index[docid] = i
                

    def __call__(
        self, query: str, k: int = 10, threads: int = 16,
    ) -> Union[list[str], list[dotdict]]:
        hits = self.searcher.search(query, k=k, threads=threads)
        
        topk = []
        for rank, hit in enumerate(hits, start=1):
            if self.dataset is not None:
                row = self.dataset_id_to_index[hit.docid]
                text = ' '.join(self.dataset[field][row] for field in self.text_fields)
                pid = self.dataset[self.id_field][row]
            else:
                # Pyserini prebuilt faiss indexes can perform docid lookup
                psg = json.loads(self.searcher.doc(hit.docid).raw())
                text = ' '.join(psg[field] for field in self.text_fields)
                pid = psg[self.id_field]
            
            topk.append({
                'text': text,
                'long_text': text,
                'pid': pid,
                'score': hit.score,
                'rank': rank,
            })
        
        return [dotdict(psg) for psg in topk]