from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import *
from dspy.utils.caching import *
from dspy.utils.logging_utils import *

import os
import ujson
import requests

def download(url):
    filename = os.path.basename(url)
    remote_size = int(requests.head(url, allow_redirects=True).headers.get('Content-Length', 0))
    local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    if local_size != remote_size:
        print(f"Downloading '{filename}'...")
        with requests.get(url, stream=True) as r, open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
