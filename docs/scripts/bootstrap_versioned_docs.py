#!/usr/bin/env python3
"""Reconstruct all configured historical documentation snapshots atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from publish_versioned_docs import DEFAULT_BRANCH

OPTIMIZER_REQUIREMENTS = ("htmlmin2==0.1.13",)


def run(*command: object, cwd: Path | None = None) -> None:
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def output(*command: object, cwd: Path | None = None) -> str:
    return subprocess.check_output([str(part) for part in command], cwd=cwd, text=True).strip()


def download_pypi_wheel(version: str, destination: Path) -> Path:
    with urlopen(f"https://pypi.org/pypi/dspy/{version}/json") as response:
        release = json.load(response)
    wheels = [item for item in release["urls"] if item["packagetype"] == "bdist_wheel"]
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel for dspy=={version}, found {len(wheels)}")
    wheel = wheels[0]
    path = destination / wheel["filename"]
    with urlopen(wheel["url"]) as response:
        path.write_bytes(response.read())
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != wheel["digests"]["sha256"]:
        raise RuntimeError(f"PyPI digest mismatch for {path.name}")
    return path


def set_release_metadata(worktree: Path, version: str) -> None:
    replacements = (
        (worktree / "pyproject.toml", r'(?m)^(version\s*=\s*)"[^"]+"', rf'\g<1>"{version}"'),
        (worktree / "dspy" / "__metadata__.py", r'(?m)^(__version__\s*=\s*)"[^"]+"', rf'\g<1>"{version}"'),
        (worktree / "pyproject.toml", r'(?m)^(name\s*=\s*)"[^"]+"', r'\g<1>"dspy"'),
        (worktree / "dspy" / "__metadata__.py", r'(?m)^(__name__\s*=\s*)"[^"]+"', r'\g<1>"dspy"'),
    )
    for path, pattern, replacement in replacements:
        text, count = re.subn(pattern, replacement, path.read_text(), count=1)
        if count != 1:
            raise RuntimeError(f"release metadata marker not found in {path}")
        path.write_text(text)


def build_tag_wheel(worktree: Path, python: Path, version: str, destination: Path, tag_date: str) -> Path:
    set_release_metadata(worktree, version)
    run(
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "--exclude-newer",
        tag_date,
        "build",
        "setuptools",
        "wheel",
        cwd=worktree,
    )
    run(
        python,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        destination,
        cwd=worktree,
    )
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one tag-built wheel for {version}, found {len(wheels)}")
    return wheels[0]


def filtered_requirements(source: Path, destination: Path) -> None:
    lines = [line for line in source.read_text().splitlines() if "github.com/stanfordnlp/dspy" not in line]
    destination.write_text("\n".join(lines) + "\n")


def bootstrap_release(
    *,
    repository: Path,
    deployment: Path,
    scripts: Path,
    release: dict[str, str],
    branch: str,
    temporary: Path,
) -> None:
    version = release["version"]
    worktree = temporary / f"source-{version}"
    environment = temporary / f"venv-{version}"
    artifact_dir = temporary / f"artifacts-{version}"
    site = temporary / f"site-{version}"
    artifact_dir.mkdir()
    run("git", "worktree", "add", "--detach", worktree, version, cwd=repository)
    try:
        run("uv", "venv", "--python", sys.executable, environment)
        python = environment / "bin" / "python"
        requirements = temporary / f"requirements-{version}.txt"
        filtered_requirements(worktree / "docs" / "requirements.txt", requirements)
        tag_date = output("git", "show", "-s", "--format=%cI", f"{version}^{{commit}}", cwd=repository)
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--exclude-newer",
            tag_date,
            "-r",
            requirements,
        )
        run("uv", "pip", "install", "--python", python, *OPTIMIZER_REQUIREMENTS)

        if release["package_source"] == "pypi-wheel":
            artifact = download_pypi_wheel(version, artifact_dir)
        else:
            artifact = build_tag_wheel(worktree, python, version, artifact_dir, tag_date)
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--exclude-newer",
            tag_date,
            artifact,
        )
        run(
            python,
            scripts / "build_docs.py",
            "release",
            "--config",
            worktree / "docs" / "mkdocs.yml",
            "--output",
            site,
            "--version",
            version,
            "--artifact",
            artifact,
            "--package-source",
            release["package_source"],
            "--renderer",
            release["renderer"],
        )
        run(
            sys.executable,
            scripts / "publish_versioned_docs.py",
            "--repository",
            deployment,
            "--site",
            site,
            "--identifier",
            version,
            "--alias",
            ".".join(version.split(".")[:2]),
            "--renderer",
            release["renderer"],
            "--package-source",
            release["package_source"],
            "--branch",
            branch,
        )
    finally:
        run("git", "worktree", "remove", "--force", worktree, cwd=repository)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("docs/versioning/releases.json"))
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    repository = args.repository.resolve()
    deployment = args.deployment.resolve()
    manifest = json.loads(args.manifest.resolve().read_text())
    releases = [release for release in manifest["releases"] if not args.only or release["version"] in args.only]
    unknown = set(args.only) - {release["version"] for release in releases}
    if unknown:
        parser.error(f"versions absent from manifest: {', '.join(sorted(unknown))}")

    scripts = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="dspy-docs-bootstrap-") as directory:
        temporary = Path(directory)
        for release in releases:
            bootstrap_release(
                repository=repository,
                deployment=deployment,
                scripts=scripts,
                release=release,
                branch=args.branch,
                temporary=temporary,
            )


if __name__ == "__main__":
    main()
