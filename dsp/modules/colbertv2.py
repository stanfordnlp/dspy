import functools
from typing import Any, Optional, Union, List

import requests
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict
import os

# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2:
    """Wrapper for the ColBERTv2 Retrieval."""

    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: Optional[Union[str, int]] = None,
        post_requests: bool = False,
    ):
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def __call__(
        self, query: str, k: int = 10, simplify: bool = False,
    ) -> Union[list[str], list[dotdict]]:
        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@CacheMemory.cache
def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert (
        k <= 100
    ), "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)

    topk = res.json()["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


@functools.cache
@NotebookCacheMemory.cache
def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    return colbertv2_get_request_v2(*args, **kwargs)


colbertv2_get_request = colbertv2_get_request_v2_wrapped


@CacheMemory.cache
def colbertv2_post_request_v2(url: str, query: str, k: int):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    res = requests.post(url, json=payload, headers=headers, timeout=10)

    return res.json()["topk"][:k]


@functools.cache
@NotebookCacheMemory.cache
def colbertv2_post_request_v2_wrapped(*args, **kwargs):
    return colbertv2_post_request_v2(*args, **kwargs)


colbertv2_post_request = colbertv2_post_request_v2_wrapped
os.environ['COLBERT_LOAD_TORCH_EXTENSION_VERBOSE'] = "True"

class ColBERTv2Local:
    def __init__(self,checkpoint:str='colbert-ir/colbertv2.0',passages:List[str]=[],index_name_or_path:str = "Colbert-RM",experiment_name:str="Colbert-Experiment",load_only:bool=False,nranks:int=1,nbits:int=2,DOC_MAXLEN:int=300,INDEX_BSIZE:int=256,KMEANS_ITER:int=8):


        self.checkpoint = checkpoint
        self.index_name_or_path = index_name_or_path
        self.experiment_name = experiment_name
        self.nranks = nranks
        self.nbits = nbits
        self.DOC_MAXLEN = DOC_MAXLEN
        self.INDEX_BSIZE = INDEX_BSIZE
        self.KMEANS_ITER = KMEANS_ITER
        self.passages = passages

        if not load_only:
            print(f"Building the index for experiment {self.experiment_name} with index name {self.index_name_or_path}")
            self.build_index()
        
        print(f"Loading the index for experiment {self.experiment_name} with index name {self.index_name_or_path}")
        self.searcher = self.get_index()

    def build_index(self):

        try:
            import colbert
        except ImportError:
            print("Colbert not found. Please check your installation or install the module using pip install colbert-ai[faiss-gpu,torch].")

        from colbert import Indexer
        from colbert.infra import Run, RunConfig, ColBERTConfig
        with Run().context(RunConfig(nranks=self.nranks, experiment=self.experiment_name)):  
            config = ColBERTConfig(doc_maxlen=self.DOC_MAXLEN, nbits=self.nbits, kmeans_niters=self.KMEANS_ITER,index_bsize=self.INDEX_BSIZE)
                                                                                       

            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            indexer.index(name=self.index_name_or_path, collection=self.passages, overwrite=True)

    def get_index(self):
        try:
            import colbert
        except ImportError:
            print("Colbert not found. Please check your installation or install the module using pip install colbert-ai[faiss-gpu,torch].")

        from colbert import Searcher
        from colbert.infra import Run, RunConfig
        
        with Run().context(RunConfig(experiment=self.experiment_name)):
            searcher = Searcher(index=self.index_name_or_path, collection=self.passages)
        return searcher
    
    def __call__(self,query:str,k:int=7,**kwargs):
        try:
            import colbert
        except ImportError:
            print("Colbert not found. Please check your installation or install the module using pip install colbert-ai[faiss-gpu,torch].")
        import torch
        
        if kwargs.get("filtered_pids"):
            filtered_pids = kwargs.get("filtered_pids")
            assert type(filtered_pids) == List[int], "The filtered pids should be a list of integers"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = self.searcher.search(
                query,
                #Number of passages to receive
                k=k, 
                #Passing the filter function of relevant 
                filter_fn=lambda pids: torch.tensor(
                    [pid for pid in pids if pid in filtered_pids],dtype=torch.int32).to(device))
            
        return results