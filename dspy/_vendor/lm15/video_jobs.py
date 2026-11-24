"""Explicit ticket handles over video_submit/status/result/list.

Snapshot properties never perform I/O. Waiting never submits or retries a failed
request, and abandoning a wait does not cancel provider generation or billing.
See the contract's 2026-09-11-shared-handles-and-profile-migration decision.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

from .types import VideoJobInfo, VideoPart, VideoStatus

if TYPE_CHECKING:
    from .providers.async_base import AsyncBaseProviderLM
    from .providers.base import BaseProviderLM

__all__ = ["AsyncVideoJob", "VideoJob", "VideoWaitTimeout"]


class VideoWaitTimeout(TimeoutError):
    """Local polling deadline, not an HTTP timeout; the job may still be running."""

    def __init__(self, info: VideoJobInfo):
        self.info = info
        super().__init__(f"video {info.id} still {info.status!r} when the wait deadline expired")


def _validate_wait(poll_every: float, timeout: float | None) -> None:
    for name, value, positive in (("poll_every", poll_every, True), ("timeout", timeout, False)):
        if value is None and name == "timeout":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")


class _Snapshot:
    _info: VideoJobInfo

    @property
    def info(self) -> VideoJobInfo:
        """The frozen snapshot from the last successful provider contact."""
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
        return f"{type(self).__name__}(id={self.id!r}, status={self.status!r}, progress={self.progress!r})"


class VideoJob(_Snapshot):
    def __init__(self, lm: BaseProviderLM, info: VideoJobInfo) -> None:
        self._lm, self._info = lm, info

    def refresh(self) -> VideoJob:
        self._info = self._lm.video_status(self.id)
        return self

    def wait(self, poll_every: float = 5.0, timeout: float | None = 300.0) -> VideoJob:
        """Wait for a terminal status, including failure; inspect status afterward.

        timeout=None deliberately waits indefinitely. The sync deadline bounds
        polling, not an in-flight status call: its transport timeout still applies.
        """
        _validate_wait(poll_every, timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.info.done:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise VideoWaitTimeout(self.info)
            time.sleep(poll_every if remaining is None else min(poll_every, remaining))
            if deadline is not None and time.monotonic() >= deadline:
                raise VideoWaitTimeout(self.info)
            self.refresh()
        return self

    def result(self) -> VideoPart:
        """Fetch through the raw result operation; never implicitly wait."""
        return self._lm.video_result(self.id)


class AsyncVideoJob(_Snapshot):
    def __init__(self, lm: AsyncBaseProviderLM, info: VideoJobInfo) -> None:
        self._lm, self._info = lm, info

    async def refresh(self) -> AsyncVideoJob:
        self._info = await self._lm.video_status(self.id)
        return self

    async def wait(self, poll_every: float = 5.0, timeout: float | None = 300.0) -> AsyncVideoJob:
        """Native cancellable wait. Neither cancellation nor timeout cancels the job."""
        from .transports._timeouts import wait_for

        _validate_wait(poll_every, timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.info.done:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise VideoWaitTimeout(self.info)
            await asyncio.sleep(poll_every if remaining is None else min(poll_every, remaining))
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise VideoWaitTimeout(self.info)
            # Return provider exceptions as data so a provider's own TimeoutError
            # is never mistaken for the local wait deadline.
            async def status():
                try:
                    return await self._lm.video_status(self.id), None
                except Exception as exc:
                    return None, exc
            try:
                info, error = await wait_for(status(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise VideoWaitTimeout(self.info) from exc
            if error is not None:
                raise error
            self._info = info
        return self

    async def result(self) -> VideoPart:
        return await self._lm.video_result(self.id)
