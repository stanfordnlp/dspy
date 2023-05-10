import os
import pickle as pkl
import json
from pathlib import Path

from dsp.modules.colbertv2 import colbertv2_get_request_v2_wrapped

from joblib.memory import MemorizedFunc


cachedir = os.environ.get('DSP_CACHEDIR') or os.path.join(Path.home(), 'cachedir_joblib') # assuming this points up to dsp/modules

def load_from_cache(dir):
    res = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".pkl"):

                pkl_path = os.path.join(root, file)
                metadata_path = os.path.join(root, "metadata.json")

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as meta_f:
                        metadata = json.load(meta_f)

                        with open(pkl_path, "rb") as f:
                            loaded = pkl.load(f)

                            res.append((metadata["input_args"], loaded))

                else:
                    print("Metadata missing for: ", metadata_path)
    return res



if __name__ == "__main__":
    gpt3_cached = load_from_cache(os.path.join(cachedir, "colbertv2", "colbertv2_get_request_v2_wrapped"))
    colbert_cached = load_from_cache(os.path.join(cachedir, "gpt3", "cached_gpt3_request_v2_wrapped"))

