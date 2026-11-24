"""
lm15.batch — the BatchJob handle (ticket ergonomics over the pure ops).

A batch job is a ticket: the provider holds the requests, ``job.id`` is
the ticket number, and the id is a plain string you can store anywhere.
The handle is per-language sugar over five pure operations
(``batch_submit`` / ``batch_status`` / ``batch_results`` /
``batch_cancel`` / ``batch_list``); ports implement the ops, the handle
idiom is theirs to choose.

``refresh()`` / ``wait()`` / ``cancel()`` update the handle's snapshot in
place and return the handle — a mutating refresh avoids the
forgot-to-reassign-stale-status bug. The frozen-types culture applies to
canonical types (``BatchJobInfo``, ``BatchEntry``), not client handles.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from .types import BatchEntry, BatchJobInfo, BatchStatus

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .providers.async_base import AsyncBaseLM
    from .providers.base import BaseProviderLM

__all__ = ["AsyncBatchJob", "BatchJob"]

_DEFAULT_POLL_SECONDS = 30.0


class BatchJob:
    """A live handle on one provider-side batch job."""

    def __init__(self, lm: "BaseProviderLM", info: BatchJobInfo) -> None:
        self._lm = lm
        self._info = info

    # ─── Snapshot access ─────────────────────────────────────────────

    @property
    def info(self) -> BatchJobInfo:
        """The frozen snapshot from the last provider contact."""
        return self._info

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def status(self) -> BatchStatus:
        return self._info.status

    @property
    def label(self) -> str | None:
        return self._info.label

    @property
    def created_at(self) -> str | None:
        return self._info.created_at

    @property
    def provider_data(self) -> dict[str, Any] | None:
        return self._info.provider_data

    @property
    def done(self) -> bool:
        return self._info.done

    def __repr__(self) -> str:
        label = f" label={self._info.label!r}" if self._info.label else ""
        return f"BatchJob(id={self._info.id!r}, status={self._info.status!r}{label})"

    # ─── Verbs (update the snapshot in place, return self) ───────────

    def refresh(self) -> "BatchJob":
        self._info = self._lm.batch_status(self._info.id)
        return self

    def wait(self, poll_every: float = _DEFAULT_POLL_SECONDS, timeout: float | None = None) -> "BatchJob":
        """Poll until the job is terminal. Convenience for small jobs and
        notebooks; the primary pattern for real workloads is store-the-id
        and re-attach."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._info.done:
                return self
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"batch {self._info.id} still {self._info.status!r} after {timeout}s"
                )
            time.sleep(poll_every)
            self.refresh()

    def results(self) -> tuple[BatchEntry, ...]:
        return self._lm.batch_results(self._info.id)

    def cancel(self) -> "BatchJob":
        self._info = self._lm.batch_cancel(self._info.id)
        return self


class AsyncBatchJob:
    """Async twin of :class:`BatchJob`; identical types, awaitable verbs."""

    def __init__(self, lm: "AsyncBaseLM", info: BatchJobInfo) -> None:
        self._lm = lm
        self._info = info

    @property
    def info(self) -> BatchJobInfo:
        return self._info

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def status(self) -> BatchStatus:
        return self._info.status

    @property
    def label(self) -> str | None:
        return self._info.label

    @property
    def created_at(self) -> str | None:
        return self._info.created_at

    @property
    def provider_data(self) -> dict[str, Any] | None:
        return self._info.provider_data

    @property
    def done(self) -> bool:
        return self._info.done

    def __repr__(self) -> str:
        label = f" label={self._info.label!r}" if self._info.label else ""
        return f"AsyncBatchJob(id={self._info.id!r}, status={self._info.status!r}{label})"

    async def refresh(self) -> "AsyncBatchJob":
        self._info = await self._lm.batch_status(self._info.id)
        return self

    async def wait(self, poll_every: float = _DEFAULT_POLL_SECONDS, timeout: float | None = None) -> "AsyncBatchJob":
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._info.done:
                return self
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"batch {self._info.id} still {self._info.status!r} after {timeout}s"
                )
            await asyncio.sleep(poll_every)
            await self.refresh()

    async def results(self) -> tuple[BatchEntry, ...]:
        return await self._lm.batch_results(self._info.id)

    async def cancel(self) -> "AsyncBatchJob":
        self._info = await self._lm.batch_cancel(self._info.id)
        return self
