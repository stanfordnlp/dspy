"""Only prepared, platform-tagged wheels; never download during a build."""
import json
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class RuntimeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        payload = Path(__file__).parent / "dspy_mxc_runtime/payload.json"
        if not payload.exists():
            raise RuntimeError("Run prepare_assets.py first (use --target diagnostic for no executors)")
        self.payload = json.loads(payload.read_text())
        self.root_is_pure = self.payload["target"] == "diagnostic"
        # Place native data in wheel platlib, not a .data/purelib side tree.
        self.distribution.has_ext_modules = lambda: not self.root_is_pure

    def get_tag(self):
        return "py3", "none", self.payload["wheel_platform"]


setup(cmdclass={"bdist_wheel": RuntimeWheel})
