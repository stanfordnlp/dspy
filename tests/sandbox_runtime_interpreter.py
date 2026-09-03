"""The reference SUB_DSPY worker run under Anthropic's sandbox-runtime (``srt``).

``srt`` (``npm install -g @anthropic-ai/sandbox-runtime``) applies OS-level filesystem and network
policy to an arbitrary process: Seatbelt on macOS, bubblewrap + seccomp on Linux. Wrapping the
worker gives the SUB_DSPY interpreter teeth: real dspy runs inside, writes are confined to a scratch
directory, and egress is limited to ``allowed_domains`` (typically just the LM endpoint), so an
inherited API key can only be spent against those hosts. Child processes inherit the sandbox, so the
nested interpreters the worker creates are sandboxed too.
"""

import json
import os
import shutil
import sys
import tempfile
from typing import Any, Callable

from tests.sub_dspy_interpreters import SubDspySubprocessInterpreter


class SandboxRuntimeInterpreter(SubDspySubprocessInterpreter):
    def __init__(
        self,
        *,
        allowed_domains: tuple[str, ...] = (),
        allow_write: tuple[str, ...] = (),
        deny_read: tuple[str, ...] = ("~/.ssh",),
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        execution_timeout: float | None = None,
    ) -> None:
        self._scratch = tempfile.mkdtemp(prefix="dspy-srt-")
        try:
            settings = {
                "network": {"allowedDomains": list(allowed_domains), "deniedDomains": []},
                "filesystem": {
                    "denyRead": list(deny_read),
                    "allowWrite": [self._scratch, *allow_write],
                    "denyWrite": [],
                },
            }
            settings_path = os.path.join(self._scratch, "settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f)
            # srt joins its arguments into one shell command, so the worker is launched through a script.
            launcher = os.path.join(self._scratch, "python")
            with open(launcher, "w", encoding="utf-8") as f:
                f.write(f'#!/bin/sh\nexec srt -s "{settings_path}" "{sys.executable}" "$@"\n')
            os.chmod(launcher, 0o700)
            super().__init__(tools=tools, output_fields=output_fields, execution_timeout=execution_timeout, python=launcher)
        except Exception:
            shutil.rmtree(self._scratch, ignore_errors=True)
            raise

    def shutdown(self) -> None:
        super().shutdown()
        shutil.rmtree(self._scratch, ignore_errors=True)
