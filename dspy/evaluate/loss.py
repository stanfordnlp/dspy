from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from snoop import snoop


@dataclass
class Trace:
    args: tuple
    kwargs: dict
    log: list[str] = field(default_factory=list)

    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    def __enter__(self):
        self.log = []
        self.started_at = datetime.now(UTC)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.completed_at = datetime.now(UTC)

    def __str__(self):
        return "\n".join(self.log)

    def write(self, line: str):
        if self.is_complete:
            raise ValueError("Trace is complete and cannot be modified")
        self.log.append(line)


@dataclass
class Loss:
    """Expected usage:

    loss_fn = dspy.Loss(...)

    pred_loss = loss_fn(prediction, ...)
    str(loss_fn.traces[-1]) => the line by line execution trace of the last call
    """

    fn: Callable[..., float]
    watch: dict = field(default_factory=dict)
    traces: list[Trace] = field(default_factory=list)
    # do we need a list here? or is it enough to have just the last trace

    def __call__(self, *args, **kwargs) -> float:
        """Calls the underlying metric function and returns the result,
        whilst also tracing the function execution and appending it to the history.
        """
        with Trace(args=args, kwargs=kwargs) as trace:
            result = snoop(out=trace, **self.watch)(self.fn)(*args, **kwargs)
            self.traces.append(trace)
        return result
