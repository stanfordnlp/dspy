from typing import Optional, Union, List, Any
import httpx
import logging
import os
from langfuse.client import Langfuse, StatefulTraceClient, StatefulSpanClient, StateType
from dsp.trackers.base import BaseTracker


class LangfuseTracker(BaseTracker):
    log = logging.getLogger("langfuse")

    def __init__(self, *, public_key: Optional[str] = None, secret_key: Optional[str] = None,
                 host: Optional[str] = None, debug: bool = False, stateful_client: Optional[
                Union[StatefulTraceClient, StatefulSpanClient]
            ] = None, update_stateful_client: bool = False, version: Optional[str] = None,
                 session_id: Optional[str] = None, user_id: Optional[str] = None, trace_name: Optional[str] = None,
                 release: Optional[str] = None, metadata: Optional[Any] = None, tags: Optional[List[str]] = None,
                 threads: Optional[int] = None, flush_at: Optional[int] = None, flush_interval: Optional[int] = None,
                 max_retries: Optional[int] = None, timeout: Optional[int] = None, enabled: Optional[bool] = None,
                 httpx_client: Optional[httpx.Client] = None, sdk_integration: str = "default") -> None:
        super().__init__()
        self.version = version
        self.session_id = session_id
        self.user_id = user_id
        self.trace_name = trace_name
        self.release = release
        self.metadata = metadata
        self.tags = tags

        self.root_span = None
        self.update_stateful_client = update_stateful_client
        self.langfuse = None

        prio_public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        prio_secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        prio_host = host or os.environ.get(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )

        if stateful_client and isinstance(stateful_client, StatefulTraceClient):
            self.trace = stateful_client
            self._task_manager = stateful_client.task_manager
            return

        elif stateful_client and isinstance(stateful_client, StatefulSpanClient):
            self.root_span = stateful_client
            self.trace = StatefulTraceClient(
                stateful_client.client,
                stateful_client.trace_id,
                StateType.TRACE,
                stateful_client.trace_id,
                stateful_client.task_manager,
            )
            self._task_manager = stateful_client.task_manager
            return

        args = {
            "public_key": prio_public_key,
            "secret_key": prio_secret_key,
            "host": prio_host,
            "debug": debug,
        }

        if release is not None:
            args["release"] = release
        if threads is not None:
            args["threads"] = threads
        if flush_at is not None:
            args["flush_at"] = flush_at
        if flush_interval is not None:
            args["flush_interval"] = flush_interval
        if max_retries is not None:
            args["max_retries"] = max_retries
        if timeout is not None:
            args["timeout"] = timeout
        if enabled is not None:
            args["enabled"] = enabled
        if httpx_client is not None:
            args["httpx_client"] = httpx_client
        args["sdk_integration"] = sdk_integration

        self.langfuse = Langfuse(**args)
        self.trace: Optional[StatefulTraceClient] = None
        self._task_manager = self.langfuse.task_manager

    def call(self, i, o, name=None, **kwargs):
        self.langfuse.trace(input=i, output=o, name=name, metadata=kwargs)
