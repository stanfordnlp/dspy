import functools
from typing import Any, Optional, Union, List

import requests
import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
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
    def __init__(self,checkpoint:str='colbert-ir/colbertv2.0'):

        self.checkpoint = checkpoint


    def build_index(self,passages:List[str],nranks:int=1,index_name_or_path:str = "Colbert-RM-",nbits:int=2,DOC_MAXLEN:int=300,INDEX_BSIZE:int=256,KMEANS_ITER:int=8,experiment_name:str="Colbert-Experiment"):

        with Run().context(RunConfig(nranks=nranks, experiment=experiment_name)):  
            config = ColBERTConfig(doc_maxlen=DOC_MAXLEN, nbits=nbits, kmeans_niters=KMEANS_ITER,index_bsize=INDEX_BSIZE)
                                                                                       

            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            indexer.index(name=index_name_or_path, collection=passages, overwrite=True)

    def get_index(self,index_name_or_path:str = "Colbert-RM-",experiment_name:str="Colbert-Experiment",passages:List[str] = []):
        with Run().context(RunConfig(experiment=experiment_name)):
            searcher = Searcher(index=index_name_or_path, collection=passages)
        self.searcher = searcher
        return searcher
    
    def get_docs(self,searcher:Searcher,query:str,k:int=7):
        
        results = searcher.search(
            query,
            #Number of passages to receive
            k=k)
            #Passing the filter function of relevant 
            # filter_fn=lambda pids: torch.tensor(
            #     [pid for pid in pids if pid in relevant_ids],dtype=torch.int32).to(device))
        return results