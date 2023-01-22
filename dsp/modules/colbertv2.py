import functools
import requests

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict


# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2:
    def __init__(self, url='http://0.0.0.0', port=None):
        self.url = f'{url}:{port}' if port else url
        self.headers = {"Content-Type": "application/json; charset=utf-8"}
    
    def post(self, query: str, k: int):
        payload = {"query": query, "k": k}
        res = requests.post(self.url, json=payload, headers=self.headers)

        return res.json()['topk'][:k]

    def __call__(self, query: str, k=10, simplify=False):
        topk = colbertv2_get_request(self.url, query, k)
        topk = [dotdict(psg) for psg in topk]

        if simplify:
            topk = [psg.long_text for psg in topk]

        return topk


@CacheMemory.cache
def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert k <= 100, f'Only k <= 100 is supported for the hosted ColBERTv2 server at the moment.'

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload)

    topk = res.json()['topk'][:k]
    topk = [{**d, 'long_text': d['text']} for d in topk]

    return topk[:k]


@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    return colbertv2_get_request_v2(*args, **kwargs)


colbertv2_get_request = colbertv2_get_request_v2_wrapped
