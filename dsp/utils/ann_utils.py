from typing import Tuple

try:
    import faiss
    from faiss import Index
except ImportError:
    raise ImportError(
        "You need to install FAISS library to perform ANN/KNN. Please check the official doc: "
        "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md",
    )


def determine_devices(max_gpu_devices: int = 0) -> Tuple[int, bool]:
    """
    Determine which device we should use
    Args:
        max_gpu_devices: an integer value, define how many GPUs we'll use.
            -1 means all devices. 0 means there are no GPUs. Default is 0.

    Returns: number of devices and is it allowed to use CUDA device (True if yes)
    """
    n_devices_total = faiss.get_num_gpus()
    is_gpu = n_devices_total > 0

    if max_gpu_devices > 0 and is_gpu:
        num_devices = min(n_devices_total, max_gpu_devices)
    elif max_gpu_devices == -1 and is_gpu:
        num_devices = n_devices_total
    else:
        num_devices = 1
        is_gpu = False
    return num_devices, is_gpu


def _get_brute_index(emb_dim: int, dist_type: str) -> Index:
    if dist_type.lower() == 'ip':
        index = faiss.IndexFlatIP(emb_dim)
    elif dist_type.lower() == 'l2':
        index = faiss.IndexFlatL2(emb_dim)
    else:
        raise ValueError(f'Wrong distance type for FAISS Flat Index: {dist_type}')

    return index


def _get_ivf_index(
    emb_dim: int,
    n_objects: int,
    in_list_dist_type: str,
    centroid_dist_type: str,
    encode_residuals: bool,
) -> Index:
    # according to the FAISS doc, this should be OK
    n_list = int(4 * (n_objects ** 0.5))

    if in_list_dist_type.lower() == 'ip':
        quannizer = faiss.IndexFlatIP(emb_dim)
    elif in_list_dist_type.lower() == 'l2':
        quannizer = faiss.IndexFlatL2(emb_dim)
    else:
        raise ValueError(f'Wrong distance type for FAISS quantizer: {in_list_dist_type}')

    if centroid_dist_type.lower() == 'ip':
        centroid_metric = faiss.METRIC_INNER_PRODUCT
    elif centroid_dist_type.lower() == 'l2':
        centroid_metric = faiss.METRIC_L2
    else:
        raise ValueError(f'Wrong distance type for FAISS index: {centroid_dist_type}')

    index = faiss.IndexIVFScalarQuantizer(
        quannizer,
        emb_dim,
        n_list,
        faiss.ScalarQuantizer.QT_fp16,  # TODO: should be optional?
        centroid_metric,
        encode_residuals,
    )
    return index


def create_faiss_index(
    emb_dim: int,
    n_objects: int,
    n_probe: int = 10,
    max_gpu_devices: int = 0,
    encode_residuals: bool = True,
    in_list_dist_type: str = 'L2',
    centroid_dist_type: str = 'L2',
) -> Index:
    """
    Create IVF index (with IP or L2 dist), without adding data and training
    Args:
        emb_dim: size of each embedding
        n_objects: size of a trainset for index. Used to determine optimal type
            of index and its settings (will use bruteforce if `n_objects` is less than 20_000).
        n_probe: number of closest IVF-clusters to check for neighbours.
            Doesn't affect bruteforce-based search.
        max_gpu_devices: maximum amount of GPUs to use for ANN-index. 0 if run on CPU.
        encode_residuals: whether or not compute residuals. The residual vector is 
            the difference between a vector and the reconstruction that can be
            decoded from its representation in the index.
        in_list_dist_type: type of distance to calculate simmilarities within one IVF.
            Can be `IP` (for inner product) or `L2` distance. Case insensetive.
            If the index type is bruteforce (`n_objects` < 20_000), this variable will define
            the distane type for that bruteforce index. `centroid_dist_type` will be ignored.
        centroid_dist_type: type of distance to calculate simmilarities between a query 
            and cluster centroids. Can be `IP` (for inner product) or `L2` distance.
            Case insensetive.
    Returns: untrained FAISS-index
    """
    if n_objects < 20_000:
        # if less than 20_000 / (4 * sqrt(20_000)) ~= 35 points per cluster - make bruteforce
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-below-1m-vectors-ivfk
        index = _get_brute_index(emb_dim=emb_dim, dist_type=in_list_dist_type)
    else:
        index = _get_ivf_index(
            emb_dim=emb_dim,
            n_objects=n_objects,
            in_list_dist_type=in_list_dist_type,
            centroid_dist_type=centroid_dist_type,
            encode_residuals=encode_residuals,
        )

    index.nprobe = n_probe

    num_devices, is_gpu = determine_devices(max_gpu_devices)
    if is_gpu:
        cloner_options = faiss.GpuMultipleClonerOptions()
        cloner_options.shard = True  # split (not replicate) one index between GPUs
        index = faiss.index_cpu_to_gpus_list(index, cloner_options, list(range(num_devices)))

    return index
