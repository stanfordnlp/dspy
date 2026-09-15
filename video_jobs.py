"""
lm15.video_jobs — the VideoJob handle (ticket ergonomics over the pure ops).

A video job is a ticket, exactly like a batch job: the provider cooks,
``job.id`` is the ticket number, and the id is a plain string you can
store anywhere.  On xAI it is the ONLY copy — the wire has no list
endpoint — so the primary pattern is store-the-id and re-attach.

The handle is per-language sugar over four pure operations
(``video_submit`` / ``video_status`` / ``video_result`` /
``video_list``); ports implement the ops, the handle idiom is theirs.

``refresh()`` / ``wait()`` update the handle's snapshot in place and
return the handle — the batch precedent (a mutating refresh avoids the
forgot-to-reassign-stale-status bug).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .types import VideoJobInfo, VideoPart, VideoStatus

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .providers.async_base import AsyncBaseProviderLM
    from .providers.base import BaseProviderLM

__all__ = ["AsyncVideoJob", "VideoJob"]

_DEFAULT_POLL_SECONDS = 5.0  # video jobs finish in seconds-to-minutes, not hours


class VideoJob:
    """A live handle on one provider-side video job."""

    def __init__(self, lm: "BaseProviderLM", info: VideoJobInfo) -> None:
        self._lm = lm
        self._info = info

    # ─── Snapshot access ─────────────────────────────────────────────

    @property
    def info(self) -> VideoJobInfo:
        """The frozen snapshot from the last provider contact."""
        return self._info

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def status(self) -> VideoStatus:
        return self._info.status

    @property
    def progress(self) -> int | None:
        return self._info.progress

    def __repr__(self) -> str:
        return f"VideoJob(id={self._info.id!r}, status={self._info.status!r}, progress={self._info.progress!r})"

    # ─── Operations ──────────────────────────────────────────────────

    def refresh(self) -> "VideoJob":
        self._info = self._lm.video_status(self._info.id)
        return self

    def wait(self, poll_every: float = _DEFAULT_POLL_SECONDS, timeout: float | None = None) -> "VideoJob":
        """Poll until the job is terminal; returns the handle (check
        ``status`` — ``failed`` returns, it does not raise, mirroring
        batch's entry-level honesty)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._info.done:
                return self
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"video {self._info.id} still {self._info.status!r} after {timeout}s"
                )
            time.sleep(poll_every)
            self.refresh()

    def result(self) -> VideoPart:
        """The finished video as a VideoPart (URL- or bytes-addressed,
        the provider's own delivery mode)."""
        return self._lm.video_result(self._info.id)


class AsyncVideoJob:
    """Async mirror of :class:`VideoJob`."""

    def __init__(self, lm: "AsyncBaseProviderLM", info: VideoJobInfo) -> None:
        self._lm = lm
        self._info = info

    @property
    def info(self) -> VideoJobInfo:
        return self._info

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def status(self) -> VideoStatus:
        return self._info.status

    @property
    def progress(self) -> int | None:
        return self._info.progress

    def __repr__(self) -> str:
        return f"AsyncVideoJob(id={self._info.id!r}, status={self._info.status!r}, progress={self._info.progress!r})"

    async def refresh(self) -> "AsyncVideoJob":
        self._info = await self._lm.video_status(self._info.id)
        return self

    async def wait(self, poll_every: float = _DEFAULT_POLL_SECONDS, timeout: float | None = None) -> "AsyncVideoJob":
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._info.done:
                return self
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"video {self._info.id} still {self._info.status!r} after {timeout}s"
                )
            await asyncio.sleep(poll_every)
            await self.refresh()

    async def result(self) -> VideoPart:
        return await self._lm.video_result(self._info.id)
