from typing import Any, Union, Optional, List
from dspy.primitives.prediction import Prediction
import dspy
from dsp.utils import dotdict
import requests
import functools

class ColBERTv2(dspy.Retrieve):
    def __init__(self, url: str = "http://0.0.0.0", port: Optional[Union[str, int]] = None, post_requests: bool = False):
        super().__init__(embedder=None)
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def forward(self, query: str, k: int = 10) -> Any:
        if self.post_requests:
            topk = colbertv2_post_request(self.url, query, k)
        else:
            topk = colbertv2_get_request(self.url, query, k)
        return dotdict({'passages': [dotdict(psg) for psg in topk]})
    

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict
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