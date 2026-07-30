import os

import httpx

from dspy.streaming.messages import StatusMessage, StatusMessageProvider
from dspy.utils import exceptions
from dspy.utils.annotation import experimental
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM, DummyVectorizer, dummy_rm
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.syncify import syncify


def download(url):
    filename = os.path.basename(url)
    remote_size = int(httpx.head(url, follow_redirects=True).headers.get("Content-Length", 0))
    local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    if not os.path.exists(filename) or local_size != remote_size:
        print(f"Downloading '{filename}'...")
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r, open(filename, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
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
