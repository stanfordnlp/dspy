import os

import requests

from dspy.streaming.messages import StatusMessage, StatusMessageProvider
from dspy.utils import exceptions
from dspy.utils.annotation import experimental
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM, DummyVectorizer, dummy_rm
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.syncify import syncify


def download(url):
    filename = os.path.basename(url)
    remote_size = int(requests.head(url, allow_redirects=True).headers.get("Content-Length", 0))
    local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    if local_size != remote_size:
        print(f"Downloading '{filename}'...")
        with requests.get(url, stream=True) as r, open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


__all__ = [
    "download",
    "exceptions",
    "BaseCallback",
    "with_callbacks",
    "DummyLM",
    "DummyVectorizer",
    "dummy_rm",
    "experimental",
    "StatusMessage",
    "StatusMessageProvider",
    "pretty_print_history",
]
