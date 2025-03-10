import s3fs

from ..utils.deprecation_utils import deprecated


@deprecated("Use s3fs.S3FileSystem instead.")
class S3FileSystem(s3fs.S3FileSystem):
    """
    `datasets.filesystems.S3FileSystem` is a subclass of [`s3fs.S3FileSystem`](https://s3fs.readthedocs.io/en/latest/api.html).

    Users can use this class to access S3 as if it were a file system. It exposes a filesystem-like API (ls, cp, open, etc.) on top of S3 storage. Provide credentials either explicitly (`key=`, `secret=`) or with boto's credential methods. See botocore documentation for more information. If no credentials are available, use `anon=True`.

    Args:
        anon (`bool`, default to `False`):
            Whether to use anonymous connection (public buckets only). If `False`, uses the key/secret given,
            or boto's credential resolver (client_kwargs, environment, variables, config files, EC2 IAM server, in that order).
        key (`str`):
            If not anonymous, use this access key ID, if specified.
        secret (`str`):
            If not anonymous, use this secret access key, if specified.
        token (`str`):
            If not anonymous, use this security token, if specified.
        use_ssl (`bool`, defaults to `True`):
            Whether to use SSL in connections to S3; may be faster without, but insecure. If `use_ssl` is
            also set in `client_kwargs`, the value set in `client_kwargs` will take priority.
        s3_additional_kwargs (`dict`):
            Parameters that are used when calling S3 API methods. Typically used for things
            like ServerSideEncryption.
        client_kwargs (`dict`):
            Parameters for the botocore client.
        requester_pays (`bool`, defaults to `False`):
            Whether `RequesterPays` buckets are supported.
        default_block_size (`int`):
            If given, the default block size value used for `open()`, if no specific value is given at all time.
            The built-in default is 5MB.
        default_fill_cache (`bool`, defaults to `True`):
            Whether to use cache filling with open by default. Refer to `S3File.open`.
        default_cache_type (`str`, defaults to `bytes`):
            If given, the default `cache_type` value used for `open()`. Set to `none` if no
            caching is desired. See fsspec's documentation for other available `cache_type` values.
        version_aware (`bool`, defaults to `False`):
            Whether to support bucket versioning. If enable this will require the user to have
            the necessary IAM permissions for dealing with versioned objects.
        cache_regions (`bool`, defaults to `False`):
            Whether to cache bucket regions. Whenever a new bucket is used, it will
            first find out which region it belongs to and then use the client for that region.
        asynchronous (`bool`, defaults to `False`):
            Whether this instance is to be used from inside coroutines.
        config_kwargs (`dict`):
            Parameters passed to `botocore.client.Config`.
        **kwargs:
            Other parameters for core session.
        session (`aiobotocore.session.AioSession`):
            Session to be used for all connections. This session will be used inplace of creating
            a new session inside S3FileSystem. For example: `aiobotocore.session.AioSession(profile='test_user')`.
        skip_instance_cache (`bool`):
            Control reuse of instances. Passed on to `fsspec`.
        use_listings_cache (`bool`):
            Control reuse of directory listings. Passed on to `fsspec`.
        listings_expiry_time (`int` or `float`):
            Control reuse of directory listings. Passed on to `fsspec`.
        max_paths (`int`): Control reuse of directory listings. Passed on to `fsspec`.

    Examples:

    Listing files from public S3 bucket.

    ```py
    >>> import datasets
    >>> s3 = datasets.filesystems.S3FileSystem(anon=True)  # doctest: +SKIP
    >>> s3.ls('public-datasets/imdb/train')  # doctest: +SKIP
    ['dataset_info.json.json','dataset.arrow','state.json']
    ```

    Listing files from private S3 bucket using `aws_access_key_id` and `aws_secret_access_key`.

    ```py
    >>> import datasets
    >>> s3 = datasets.filesystems.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)  # doctest: +SKIP
    >>> s3.ls('my-private-datasets/imdb/train')  # doctest: +SKIP
    ['dataset_info.json.json','dataset.arrow','state.json']
    ```

    Using `S3Filesystem` with `botocore.session.Session` and custom `aws_profile`.

    ```py
    >>> import botocore
    >>> from datasets.filesystems import S3Filesystem

    >>> s3_session = botocore.session.Session(profile_name='my_profile_name')
    >>> s3 = S3FileSystem(session=s3_session)  # doctest: +SKIP
    ```

    Loading dataset from S3 using `S3Filesystem` and [`load_from_disk`].

    ```py
    >>> from datasets import load_from_disk
    >>> from datasets.filesystems import S3Filesystem

    >>> s3 = S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)  # doctest: +SKIP
    >>> dataset = load_from_disk('s3://my-private-datasets/imdb/train', storage_options=s3.storage_options)  # doctest: +SKIP
    >>> print(len(dataset))
    25000
    ```

    Saving dataset to S3 using `S3Filesystem` and [`Dataset.save_to_disk`].

    ```py
    >>> from datasets import load_dataset
    >>> from datasets.filesystems import S3Filesystem

    >>> dataset = load_dataset("imdb")
    >>> s3 = S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)  # doctest: +SKIP
    >>> dataset.save_to_disk('s3://my-private-datasets/imdb/train', storage_options=s3.storage_options)  # doctest: +SKIP
    ```
    """
