from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any


class _TestableThread(threading.Thread):
    def __init__(self, target: Callable[..., Any], args: tuple):
        threading.Thread.__init__(self, target=target, args=args)
        self.exc: BaseException | None = None

    def run(self) -> None:
        try:
            threading.Thread.run(self)
        except BaseException as e:
            self.exc = e

    def join(self, timeout: float | None = None) -> None:
        super(_TestableThread, self).join(timeout)
        if self.exc:
            raise self.exc
