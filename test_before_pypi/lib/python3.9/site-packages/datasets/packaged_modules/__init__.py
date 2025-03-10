import inspect
import re
from typing import Dict, List, Tuple

from huggingface_hub.utils import insecure_hashlib

from .arrow import arrow
from .audiofolder import audiofolder
from .cache import cache
from .csv import csv
from .imagefolder import imagefolder
from .json import json
from .pandas import pandas
from .parquet import parquet
from .sql import sql
from .text import text
from .webdataset import webdataset


def _hash_python_lines(lines: List[str]) -> str:
    filtered_lines = []
    for line in lines:
        line = re.sub(r"#.*", "", line)  # remove comments
        if line:
            filtered_lines.append(line)
    full_str = "\n".join(filtered_lines)

    # Make a hash from all this code
    full_bytes = full_str.encode("utf-8")
    return insecure_hashlib.sha256(full_bytes).hexdigest()


# get importable module names and hash for caching
_PACKAGED_DATASETS_MODULES = {
    "csv": (csv.__name__, _hash_python_lines(inspect.getsource(csv).splitlines())),
    "json": (json.__name__, _hash_python_lines(inspect.getsource(json).splitlines())),
    "pandas": (pandas.__name__, _hash_python_lines(inspect.getsource(pandas).splitlines())),
    "parquet": (parquet.__name__, _hash_python_lines(inspect.getsource(parquet).splitlines())),
    "arrow": (arrow.__name__, _hash_python_lines(inspect.getsource(arrow).splitlines())),
    "text": (text.__name__, _hash_python_lines(inspect.getsource(text).splitlines())),
    "imagefolder": (imagefolder.__name__, _hash_python_lines(inspect.getsource(imagefolder).splitlines())),
    "audiofolder": (audiofolder.__name__, _hash_python_lines(inspect.getsource(audiofolder).splitlines())),
    "webdataset": (webdataset.__name__, _hash_python_lines(inspect.getsource(webdataset).splitlines())),
}

# get importable module names and hash for caching
_PACKAGED_DATASETS_MODULES_2_15_HASHES = {
    "csv": "eea64c71ca8b46dd3f537ed218fc9bf495d5707789152eb2764f5c78fa66d59d",
    "json": "8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96",
    "pandas": "3ac4ffc4563c796122ef66899b9485a3f1a977553e2d2a8a318c72b8cc6f2202",
    "parquet": "ca31c69184d9832faed373922c2acccec0b13a0bb5bbbe19371385c3ff26f1d1",
    "arrow": "74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137",
    "text": "c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34",
    "imagefolder": "7b7ce5247a942be131d49ad4f3de5866083399a0f250901bd8dc202f8c5f7ce5",
    "audiofolder": "d3c1655c66c8f72e4efb5c79e952975fa6e2ce538473a6890241ddbddee9071c",
}

# Used to infer the module to use based on the data files extensions
_EXTENSION_TO_MODULE: Dict[str, Tuple[str, dict]] = {
    ".csv": ("csv", {}),
    ".tsv": ("csv", {"sep": "\t"}),
    ".json": ("json", {}),
    ".jsonl": ("json", {}),
    ".parquet": ("parquet", {}),
    ".geoparquet": ("parquet", {}),
    ".gpq": ("parquet", {}),
    ".arrow": ("arrow", {}),
    ".txt": ("text", {}),
    ".tar": ("webdataset", {}),
}
_EXTENSION_TO_MODULE.update({ext: ("imagefolder", {}) for ext in imagefolder.ImageFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("imagefolder", {}) for ext in imagefolder.ImageFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext: ("audiofolder", {}) for ext in audiofolder.AudioFolder.EXTENSIONS})
_EXTENSION_TO_MODULE.update({ext.upper(): ("audiofolder", {}) for ext in audiofolder.AudioFolder.EXTENSIONS})
_MODULE_SUPPORTS_METADATA = {"imagefolder", "audiofolder"}

# Used to filter data files based on extensions given a module name
_MODULE_TO_EXTENSIONS: Dict[str, List[str]] = {}
for _ext, (_module, _) in _EXTENSION_TO_MODULE.items():
    _MODULE_TO_EXTENSIONS.setdefault(_module, []).append(_ext)

for _module in _MODULE_TO_EXTENSIONS:
    _MODULE_TO_EXTENSIONS[_module].append(".zip")
