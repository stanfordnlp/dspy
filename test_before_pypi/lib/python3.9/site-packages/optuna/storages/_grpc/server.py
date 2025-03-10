from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from optuna import logging
from optuna._experimental import experimental_func
from optuna._imports import _LazyImport
from optuna.storages import BaseStorage


if TYPE_CHECKING:
    import grpc

    from optuna.storages._grpc import servicer as grpc_servicer
    from optuna.storages._grpc.auto_generated import api_pb2_grpc
else:
    grpc = _LazyImport("grpc")
    grpc_servicer = _LazyImport("optuna.storages._grpc.servicer")
    api_pb2_grpc = _LazyImport("optuna.storages._grpc.auto_generated.api_pb2_grpc")


_logger = logging.get_logger(__name__)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def make_server(
    storage: BaseStorage, host: str, port: int, thread_pool: ThreadPoolExecutor | None = None
) -> grpc.Server:
    server = grpc.server(thread_pool or ThreadPoolExecutor(max_workers=10))
    api_pb2_grpc.add_StorageServiceServicer_to_server(
        grpc_servicer.OptunaStorageProxyService(storage), server
    )  # type: ignore
    server.add_insecure_port(f"{host}:{port}")
    return server


@experimental_func("4.2.0")
def run_grpc_proxy_server(
    storage: BaseStorage,
    *,
    host: str = "localhost",
    port: int = 13000,
    thread_pool: ThreadPoolExecutor | None = None,
) -> None:
    """Run a gRPC server for the given storage URL, host, and port.

    Example:

        Run this server with the following way:

        .. code::

            from optuna.storages import run_grpc_proxy_server
            from optuna.storages import get_storage

            storage = get_storage("mysql+pymysql://<user>:<pass>@<host>/<dbname>[?<options>]")
            run_grpc_proxy_server(storage, host="localhost", port=13000)

        Please refer to the client class :class:`~optuna.storages.GrpcStorageProxy` for
        the client usage. Please use :func:`~optuna.storages.get_storage` instead of
        :class:`~optuna.storages.RDBStorage` since ``RDBStorage`` by itself does not use cache in
        process and it may cause significant slowdown.

    Args:
        storage: A storage object to proxy.
        host: Hostname to listen on.
        port: Port to listen on.
        thread_pool:
            Thread pool to use for the server. If :obj:`None`, a default thread pool
            with 10 workers will be used.
    """
    server = make_server(storage, host, port, thread_pool)
    server.start()
    _logger.info(f"Server started at {host}:{port}")
    _logger.info("Listening...")
    server.wait_for_termination()
