#!/usr/bin/env python3
"""Refresh the vendored copy of lm15 under ``dspy/_vendor/lm15``.

lm15 (https://github.com/cmpnd-ai/lm15-python) is not on PyPI, so its ``lm15/``
package is copied into the dspy tree and ships in the dspy wheel. This script is
the only supported way to change those files.

Usage:
    python scripts/update_vendored_lm15.py [REF] [--source URL] [--force]

REF is any git ref of the source repository (default: main). The script fetches
that ref, replaces ``dspy/_vendor/lm15`` with its ``lm15/`` package plus the
LICENSE, and writes ``dspy/_vendor/lm15/VENDORED`` recording the source, the
exact commit, and a digest of the copied files.

If the current vendored tree does not match its recorded digest, someone edited
it by hand; the script stops so those edits are not silently thrown away. Use
--force to overwrite anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "dspy" / "_vendor" / "lm15"
DEFAULT_SOURCE = "https://github.com/cmpnd-ai/lm15-python.git"
MARKER = "VENDORED"

# Only these files are copied; anything else in the package is reported and skipped,
# so a stray editor or cache file upstream never ends up in the dspy wheel.
KEEP_SUFFIXES = {".py"}
KEEP_NAMES = {"py.typed"}


def digest(root: Path) -> str:
    """Stable digest of every vendored file except the marker itself."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != MARKER):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def recorded_digest() -> str | None:
    marker = TARGET / MARKER
    if not marker.exists():
        return None
    for line in marker.read_text().splitlines():
        if line.startswith("digest="):
            return line.partition("=")[2].strip()
    return None


def fetch(source: str, ref: str, into: Path) -> str:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=into, check=True, capture_output=True, text=True)

    run("init", "-q")
    run("fetch", "-q", "--depth", "1", source, ref)
    run("checkout", "-q", "FETCH_HEAD")
    return run("rev-parse", "HEAD").stdout.strip()


def copy_package(src_pkg: Path, license_file: Path) -> list[Path]:
    skipped = []
    if TARGET.exists():
        shutil.rmtree(TARGET)
    for path in sorted(src_pkg.rglob("*")):
        rel = path.relative_to(src_pkg)
        if path.is_dir():
            continue
        if "__pycache__" in rel.parts:
            continue
        if path.suffix not in KEEP_SUFFIXES and path.name not in KEEP_NAMES:
            skipped.append(rel)
            continue
        dest = TARGET / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(license_file, TARGET / "LICENSE")
    return skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref", nargs="?", default="main", help="git ref in the source repo (default: main)")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"source repository (default: {DEFAULT_SOURCE})")
    parser.add_argument("--force", action="store_true", help="overwrite even if the vendored tree was edited by hand")
    args = parser.parse_args()

    if TARGET.exists():
        expected = recorded_digest()
        if expected is None and not args.force:
            print(f"{TARGET} exists but has no {MARKER} file; refusing to overwrite. Use --force.", file=sys.stderr)
            return 1
        if expected is not None and digest(TARGET) != expected and not args.force:
            print(
                f"{TARGET} differs from its recorded digest: it was edited by hand.\n"
                "Move those changes to the lm15 repository first, or re-run with --force to discard them.",
                file=sys.stderr,
            )
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp)
        commit = fetch(args.source, args.ref, checkout)
        src_pkg = checkout / "lm15"
        license_file = checkout / "LICENSE"
        if not (src_pkg / "__init__.py").exists() or not license_file.exists():
            print(
                f"{args.source}@{args.ref} does not look like lm15-python (no lm15/__init__.py or LICENSE)",
                file=sys.stderr,
            )
            return 1
        skipped = copy_package(src_pkg, license_file)

    marker_body = "\n".join(
        [
            "# Written by scripts/update_vendored_lm15.py. Do not edit files in this directory by hand.",
            f"source={args.source}",
            f"ref={args.ref}",
            f"commit={commit}",
            f"updated={datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"digest={digest(TARGET)}",
            "",
        ]
    )
    (TARGET / MARKER).write_text(marker_body)

    print(f"vendored lm15 {commit[:12]} ({args.ref}) from {args.source} into {TARGET.relative_to(REPO_ROOT)}")
    for rel in skipped:
        print(f"  skipped non-source file: {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
