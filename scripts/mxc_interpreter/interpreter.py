"""Opt-in Linux MXC prototype, not a production security boundary or public DSPy API."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from dspy.primitives import local_interpreter
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.local_interpreter import LocalInterpreter


class MXCInterpreter(LocalInterpreter):
    """Reuse LocalInterpreter's worker/protocol; change only launch and session cleanup.

    ``runtime`` is the path to MXC's native executor.
    No runtime downloads, package installs, or unsandboxed fallbacks occur here.
    The current Python environment must already contain the desired libraries.
    """

    execution_instructions = (
        "Code runs in a persistent CPython worker. Installed Python libraries are available. "
        "Read and write files in the working directory. Direct network access is disabled. "
        "Host tools and SUBMIT are global functions. Finish all threads before returning."
    )

    def __init__(self, *, runtime, bwrap="bwrap", **kwargs):
        if sys.platform != "linux":
            raise ValueError("This prototype is verified only on Linux.")
        super().__init__(**kwargs)
        self.runtime = Path(runtime).resolve(strict=True)
        self.bwrap = bwrap
        self.session = None
        self.workspace = None
        self.diagnostics = ""

    @classmethod
    def configured(cls, **kwargs):
        # Keep execution instructions visible to Flex/RLM before lazy construction.
        factory = partial(cls, **kwargs)
        factory.execution_instructions = cls.execution_instructions
        return factory

    def _spawn_worker(self):
        self.session = Path(tempfile.mkdtemp(prefix="dspy-mxc-"))
        self.workspace = self.session / "work"
        self.workspace.mkdir()
        temporary = self.session / "tmp"
        temporary.mkdir()
        worker = self.session / "worker.py"
        # Stage the existing source, rather than maintaining a sandbox-specific worker.
        shutil.copyfile(Path(local_interpreter.__file__).with_name("local_interpreter_worker.py"), worker)
        bwrap = shutil.which(self.bwrap)
        if bwrap is None:
            raise FileNotFoundError(f"Bubblewrap executable not found: {self.bwrap}")
        env = {
            "PATH": os.pathsep.join([str(Path(bwrap).parent), "/usr/local/bin", os.defpath]),
            "HOME": str(temporary),
            "TMPDIR": str(temporary),
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        python = [sys.executable, "-I", str(worker)]
        # Do not expose the checkout just because its .venv supplies packages.
        readonly = sorted({sys.prefix, sys.base_prefix, str(Path(sys.executable).resolve().parent), str(worker)})
        writable = [str(self.workspace), str(temporary)]
        policy_path = self.session / "policy.json"
        policy = {
            "version": "1.0.0",
            "containment": "bubblewrap",
            "process": {
                "commandLine": "exec " + shlex.join(python),
                "cwd": str(self.workspace),
                "env": [f"{key}={value}" for key, value in env.items()],
                # DSPy owns per-execution deadlines; no competing session timer.
                "timeout": 0,
            },
            "filesystem": {"readonlyPaths": readonly, "readwritePaths": writable, "deniedPaths": ["/etc"]},
            "network": {
                "egress": {"default": "deny"},
                "ingress": {"default": "deny", "hostLoopback": "deny"},
            },
        }
        policy_path.write_text(json.dumps(policy))
        with (self.session / "stderr.log").open("w") as errors:
            return subprocess.Popen(
                [str(self.runtime), "--config", str(policy_path)],
                cwd=self.workspace,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                start_new_session=True,
                close_fds=True,
            )

    def start(self):
        try:
            super().start()
        except BaseException as exc:
            self._ended = True
            self._kill()
            if not isinstance(exc, Exception):
                raise
            raise CodeInterpreterError(f"MXC startup failed: {exc}\n{self.diagnostics}") from exc

    def _kill(self):
        try:
            # Killing the launcher process group also tears down the PID namespace.
            # Do not depend on upstream session-timeout cleanup, or just kill Python.
            super()._kill()
        finally:
            if self.session is not None:
                errors = self.session / "stderr.log"
                if errors.exists():
                    with errors.open("rb") as stream:
                        stream.seek(max(0, errors.stat().st_size - 8192))
                        self.diagnostics = stream.read().decode(errors="replace")
                shutil.rmtree(self.session)
                self.session = None
