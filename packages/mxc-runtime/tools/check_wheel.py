"""Clean offline install and package relocation smoke test (requires uv)."""
import json
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


def main():
    wheel = Path(sys.argv[1]).resolve()
    with zipfile.ZipFile(wheel) as z:
        payload = json.loads(z.read("dspy_mxc_runtime/payload.json"))
        metadata = z.read("dspy_mxc_runtime-0.1.0a1.dist-info/METADATA").decode()
        assert "Requires-Dist:" not in metadata
        if payload["target"] == "diagnostic":
            assert not any("/bin/" in n for n in z.namelist())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = root / "env"
        venv.EnvBuilder(with_pip=False).create(env)
        python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-index", "--no-deps", str(wheel)], check=True)
        code = "import dspy_mxc_runtime; print(dspy_mxc_runtime.__file__)"
        installed = Path(subprocess.check_output([str(python), "-I", "-c", code], text=True).strip()).parent
        relocated = root / "relocated"
        relocated.mkdir()
        shutil.move(str(installed), relocated / "dspy_mxc_runtime")
        code = """
import sys, subprocess
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from dspy_mxc_runtime import executables
if sys.argv[2] == "diagnostic":
    try:
        executables()
    except RuntimeError as exc:
        assert "no compatible native payload" in str(exc)
        print(str(exc))
    else:
        raise AssertionError("diagnostic runtime succeeded")
else:
    executor, bwrap = executables()
    assert executor.is_relative_to(Path(sys.argv[1]).resolve())
    subprocess.run([str(executor), "--help"], check=True, stdout=subprocess.DEVNULL)
    if bwrap is not None:
        assert bwrap.is_relative_to(Path(sys.argv[1]).resolve())
        subprocess.run([str(bwrap), "--version"], check=True)
print("clean install + relocation OK")
"""
        subprocess.run([str(python), "-I", "-c", code, str(relocated), payload["target"]], check=True)


if __name__ == "__main__":
    main()
