from functools import partial

from huggingface_hub import hf_hub_url


hf_dataset_url = partial(hf_hub_url, repo_type="dataset")
