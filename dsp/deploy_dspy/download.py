import hashlib
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock
from huggingface_hub import repo_exists, snapshot_download

class DownloadFailedError(Exception):
    pass


MODEL_HOME = "/mnt/local_storage/.cache/huggingface/"

def is_remote_path(source_path: str) -> bool:
    scheme = urlparse(source_path).scheme
    if not scheme:
        return False
    return True


def download_to_local(source_path: str):
    """Thread-safe download from remote storage"""
    scheme = urlparse(source_path).scheme
    local_path = get_local_path(source_path)
    if not is_remote_path(source_path):
        # logger.info(f"Found local path {source_path}, skipping downloading...")
        return source_path

    elif scheme == "s3":
        download_from_s3(source_path, local_path)
    elif scheme == "gcs":
        download_from_gcs(source_path, local_path)
    else:
        raise DownloadFailedError(f"Invalid remote path: {source_path}")
    return local_path


def get_lock_path(local_path: str):
    path = Path(local_path)
    parent = path.parent
    parent.mkdir(exist_ok=True, parents=True)
    return parent / (path.name + ".lock")


def download_from_s3(remote_path: str, local_path: str):
    lock_file = get_lock_path(local_path)
    with FileLock(lock_file):
        result = subprocess.run(["aws", "s3", "sync", remote_path, local_path])
        if result.returncode != 0:
            raise DownloadFailedError(
                f"Download failed from remote storage {remote_path} with result: {result}"
            )


def download_from_gcs(remote_path: str, local_path: str):
    lock_file = get_lock_path(local_path)
    with FileLock(lock_file):
        result = subprocess.run(
            ["gcloud", "storage", "rsync", "--recursive", remote_path, local_path]
        )
        if result.returncode != 0:
            raise DownloadFailedError(
                f"Download failed from remote storage with result: {result}"
            )


def get_local_path(source_path: str):
    checkpoint_path_hash = hashlib.md5(source_path.encode()).hexdigest()
    local_path = os.path.join(MODEL_HOME, f"models--checkpoint--{checkpoint_path_hash}")
    return local_path


def safe_hf_download(model_id: str):
    """Helper function for safe download from huggingface"""
    lock_path = Path(f"{MODEL_HOME}/{model_id.replace('/', '--')}.lock").expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        model_path = snapshot_download(repo_id=model_id)
    return model_path

def download_model(model_id_or_path: str):
    """Helper function to download a model given the model id or remote path"""
    if not is_remote_path(model_id_or_path):
        if not os.path.exists(model_id_or_path):
            # Make sure to download HF models in a thread-safe way explictly
            # huggingface_hub >= 0.23.1 has introduced a race condition
            repo_exists(model_id_or_path)  # make sure it exists
            model_id_or_path = safe_hf_download(model_id_or_path)
        return model_id_or_path
    return download_to_local(model_id_or_path)