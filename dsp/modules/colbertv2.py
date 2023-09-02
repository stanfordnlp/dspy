from typing import Optional, Union, Any
import uuid
import requests

import dsp
from dsp.utils import dotdict
from dsp.utils.cache import sqlite_cache_wrapper, sqlite_cache_splitter


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
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        cache_args: dict[str, Union[str, float]] = {
            "worker_id": str(uuid.uuid4()),
            "cache_end_timerange": dsp.settings.config["experiment_start_timestamp"],
            "cache_start_timerange": dsp.settings.config["experiment_end_timestamp"],
        }

        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(
                self.url, query, k, **cache_args
            )
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(
                self.url, query, k, **cache_args
            )

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert (
        k <= 100
    ), "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)

    topk = res.json()["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    return colbertv2_get_request_v2(*args, **kwargs)


@sqlite_cache_splitter
@sqlite_cache_wrapper
def colbertv2_get_request(*args, **kwargs):
    return colbertv2_get_request_v2_wrapped(*args, **kwargs)


def colbertv2_post_request_v2(url: str, query: str, k: int):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    res = requests.post(url, json=payload, headers=headers, timeout=10)

    return res.json()["topk"][:k]


@sqlite_cache_splitter
@sqlite_cache_wrapper
def colbertv2_post_request(*args, **kwargs):
    return colbertv2_post_request_v2(*args, **kwargs)
