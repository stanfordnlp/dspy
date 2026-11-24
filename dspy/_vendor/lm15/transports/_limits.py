"""
Timeouts and connection limits — the two knobs every transport takes.

Defaults follow the provider SDKs, not general-purpose HTTP clients: a
model call that thinks for three minutes before its first byte is normal,
and a client that gives up at 60 seconds turns that into a "network
failure" and, under a retry loop, restarts the generation each time.
OpenAI, Anthropic and litellm all wait 600 seconds by default; lm15 does
the same.  Connect stays short (10 s): a host that does not answer a SYN
in ten seconds is down or unreachable, and a long wait there only delays
the error.

Every timeout is per operation, not per request: ``read`` bounds the wait
for the next byte, so a slow stream that keeps trickling never times out,
and a stalled one fails at ``read`` seconds after its last byte.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 600.0
DEFAULT_WRITE_TIMEOUT = 600.0
DEFAULT_POOL_TIMEOUT = 600.0
DEFAULT_MAX_CONNECTIONS = 100


@dataclass(frozen=True, slots=True)
class Timeouts:
    """How long to wait, in seconds, at each step of a request.

    - ``connect``: TCP + TLS (and a proxy CONNECT) to the host.
    - ``read``: the next byte of the reply — headers first, then each body
      chunk.  A model that is still thinking sends nothing, so this is the
      one to raise for slow local models and long non-streaming answers.
    - ``write``: sending the request bytes.
    - ``pool``: a free connection when ``max_connections`` are all busy.
      ``None`` waits as long as it takes.

    Every value must be positive (or ``None`` for ``pool``); a zero or
    negative timeout is a configuration mistake, not "no timeout".
    """

    connect: float = DEFAULT_CONNECT_TIMEOUT
    read: float = DEFAULT_READ_TIMEOUT
    write: float = DEFAULT_WRITE_TIMEOUT
    pool: float | None = DEFAULT_POOL_TIMEOUT

    def __post_init__(self) -> None:
        for name in ("connect", "read", "write"):
            _check_positive(name, getattr(self, name))
        if self.pool is not None:
            _check_positive("pool", self.pool)


def _check_positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(
            f"Timeouts.{name} must be a positive number of seconds, got {value!r}"
        )


def check_max_connections(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_connections must be a positive integer, got {value!r}")
    return value


def read_timeout_hint(seconds: float) -> str:
    """The tail of a read-timeout message: says whose limit fired and how
    to move it.  A user who cannot tell a client timeout from a dead
    server retries the wrong thing."""
    return (
        f"no bytes for {seconds:g}s; this is lm15's read timeout, not a server "
        f"failure — raise it with Timeouts(read=...) if the model needs longer"
    )


def pool_timeout_hint(seconds: float, max_connections: int) -> str:
    return (
        f"all {max_connections} connections were busy for {seconds:g}s; raise "
        f"max_connections, or Timeouts(pool=...) to wait longer"
    )
