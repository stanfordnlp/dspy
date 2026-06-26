"""Reusable helpers for extracting and evaluating generated Python code."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import platform
from multiprocessing.connection import Connection
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat

DEFAULT_MEM_LIMIT_BYTES = 1 << 30
DEFAULT_CPU_LIMIT_SECONDS = 17

__all__ = [
    "DEFAULT_CPU_LIMIT_SECONDS",
    "DEFAULT_MEM_LIMIT_BYTES",
    "CodeExecutionResult",
    "extract_dspy_code",
    "run_python_check",
]


class CodeExecutionResult(BaseModel):
    """Result of running generated code against a check function."""

    model_config = ConfigDict(extra="forbid")

    score: StrictFloat
    error: str | None


def _worker(
    code: str,
    test: str,
    entry_point: str,
    mem_limit_bytes: int,
    cpu_limit_seconds: int,
    conn: Connection,
) -> None:
    """Child process body: apply rlimits, exec code+test, send status."""
    try:
        import resource
        import signal

        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes)
            )
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds)
            )

        def alarm_handler(_sig: int, _frame: Any) -> None:
            raise TimeoutError("CPU alarm fired")

        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(int(cpu_limit_seconds))

        namespace: dict[str, Any] = {}
        exec(code, namespace)
        exec(test, namespace)
        check_fn = namespace.get("check")
        candidate = namespace.get(entry_point)
        if check_fn is None or candidate is None:
            conn.send(("err", f"missing check or entry_point={entry_point!r}"))
            return
        check_fn(candidate)
        conn.send(("ok", None))
    except BaseException as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def run_python_check(
    *,
    code: str,
    test: str,
    entry_point: str,
    timeout: float,
    mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
    cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
) -> CodeExecutionResult:
    """Run code plus a HumanEval-style test in a sandboxed subprocess."""
    method = "fork" if platform.system() != "Windows" else "spawn"
    ctx: Any = mp.get_context(method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker,
        args=(
            code,
            test,
            entry_point,
            mem_limit_bytes,
            cpu_limit_seconds,
            child_conn,
        ),
    )
    proc.start()
    child_conn.close()
    score = 0.0
    error: str | None = None
    try:
        if parent_conn.poll(timeout):
            try:
                status, msg = parent_conn.recv()
            except EOFError:
                status, msg = "err", "EOF from worker"
            if status == "ok":
                score = 1.0
            else:
                error = str(msg)
        else:
            error = f"timeout after {timeout}s"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        with contextlib.suppress(Exception):
            parent_conn.close()
    return CodeExecutionResult(score=score, error=error)


def extract_dspy_code(pred: Any, *, field_name: str = "code") -> str:
    """Pull Python source out of a DSPy prediction field."""
    code_field = getattr(pred, field_name, None)
    if code_field is None:
        return ""
    inner = getattr(code_field, "code", None)
    if isinstance(inner, str):
        return inner
    if isinstance(code_field, str):
        return code_field
    try:
        as_str = str(code_field)
    except Exception:
        return ""
    if as_str.startswith("code="):
        try:
            return as_str.split("=", 1)[1].strip().strip("'\"")
        except Exception:
            return as_str
    return as_str
