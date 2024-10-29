from dataclasses import dataclass
from typing import Optional, List, Any, NamedTuple
import httpx
import logging
import os
from dsp.trackers.base import BaseTracker

try:
    from langfuse.client import Langfuse
    from langfuse.decorators import observe
except ImportError or NameError:
    def observe():
        def decorator(func):
            return func

        return decorator


class LangfuseTracker(BaseTracker):
    log = logging.getLogger("langfuse")

    def __init__(self, *, public_key: Optional[str] = None, secret_key: Optional[str] = None,
                 host: Optional[str] = None, debug: bool = False, version: Optional[str] = None,
                 session_id: Optional[str] = None, user_id: Optional[str] = None, trace_name: Optional[str] = None,
                 release: Optional[str] = None, metadata: Optional[Any] = None, tags: Optional[List[str]] = None,
                 threads: Optional[int] = None, flush_at: Optional[int] = None, flush_interval: Optional[int] = None,
                 max_retries: Optional[int] = None, timeout: Optional[int] = None, enabled: Optional[bool] = None,
                 httpx_client: Optional[httpx.Client] = None, sdk_integration: str = "default") -> None:
        try:
            super().__init__()
            self.version = version
            self.session_id = session_id
            self.user_id = user_id
            self.trace_name = trace_name
            self.release = release
            self.metadata = metadata
            self.tags = tags

            self.root_span = None
            self.langfuse = None

            prio_public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
            prio_secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
            prio_host = host or os.environ.get(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            )

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
            self._task_manager = self.langfuse.task_manager
        except Exception:
            self.log.info("langfuse create fail, langfuse is not installed or configured properly error.")

    def call(self, i, o, name=None, **kwargs):
        return self.langfuse.trace(input=i, output=o, name=name, metadata=kwargs)

    def get_all_quota(self, **kwargs):
        """
        Get all quota information

        Supported parameters:
        :param page: Page number
        :param limit: Items per page
        :param user_id: User identifier
        :param name: Name
        :param session_id: Session identifier
        :param from_timestamp: Start timestamp
        :param to_timestamp: End timestamp
        :param order_by: Sort field
        :param tags: Tags (string or sequence)
        ...
        """
        response = self.langfuse.fetch_traces(**kwargs)
        traces = response.data

        if not traces:
            return UsageStatistics(
                total_cost=0.0, total_latency=0.0, trace_count=0,
                average_cost=0.0, average_latency=0.0,
                max_cost=0.0, min_cost=0.0,
                max_latency=0.0, min_latency=0.0,
                trace_metrics=[]
            )

        trace_metrics = [
            TraceMetric(
                trace_id=trace.id,
                cost=trace.total_cost,
                latency=trace.latency
            ) for trace in traces
        ]

        costs = [metric.cost for metric in trace_metrics]
        latencies = [metric.latency for metric in trace_metrics]

        return UsageStatistics(
            total_cost=sum(costs),
            total_latency=sum(latencies),
            trace_count=len(traces),
            average_cost=sum(costs) / len(traces),
            average_latency=sum(latencies) / len(traces),
            max_cost=max(costs),
            min_cost=min(costs),
            max_latency=max(latencies),
            min_latency=min(latencies),
            trace_metrics=trace_metrics
        )


class TraceMetric(NamedTuple):
    """Metrics for a single trace"""
    trace_id: str
    cost: float
    latency: float


@dataclass
class UsageStatistics:
    total_cost: float
    total_latency: float
    trace_count: int

    average_cost: float
    average_latency: float

    max_cost: float
    min_cost: float
    max_latency: float
    min_latency: float

    trace_metrics: List[TraceMetric]

    def get_top_costs(self, limit: int = 5) -> List[TraceMetric]:
        """Get the most expensive traces"""
        return sorted(self.trace_metrics, key=lambda x: x.cost, reverse=True)[:limit]

    def get_top_latencies(self, limit: int = 5) -> List[TraceMetric]:
        """Get the longest traces"""
        return sorted(self.trace_metrics, key=lambda x: x.latency, reverse=True)[:limit]