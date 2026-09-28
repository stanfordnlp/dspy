#!/usr/bin/env python3
"""Import/update lm15 using a package-only, squashed Git subtree.

Run from any directory: python scripts/update_vendored_lm15.py [REF]
Defaults to main from cmpnd-ai/lm15-python. Requires a clean checkout and
creates local commits, but never pushes. --source overrides the source URL.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFIX = "dspy/_vendor/lm15"
DEFAULT_SOURCE = "https://github.com/cmpnd-ai/lm15-python.git"
RECORD = REPO_ROOT / "dspy/_vendor/lm15-provenance.txt"
LICENSE = REPO_ROOT / "dspy/_vendor/lm15-LICENSE"


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", nargs="?", default="main")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if git("status", "--porcelain", "--untracked-files=all"):
        parser.error("Commit or stash all changes first. This command creates local commits.")
    git("symbolic-ref", "--quiet", "HEAD")
    git("var", "GIT_AUTHOR_IDENT")
    git("var", "GIT_COMMITTER_IDENT")
    before = git("rev-parse", "HEAD")

    # Full history is necessary: split commit identities must remain stable
    # between updates so subtree merge can find its previous imported ancestor.
    with tempfile.TemporaryDirectory(prefix="dspy-lm15-") as tmp:
        checkout = Path(tmp) / "source"
        git("clone", "--no-checkout", args.source, str(checkout))
        git("fetch", "origin", args.ref, cwd=checkout)
        source_commit = git("rev-parse", "FETCH_HEAD", cwd=checkout)
        git("checkout", "--detach", source_commit, cwd=checkout)
        contract = git("show", f"{source_commit}:CONTRACT_PIN", cwd=checkout)
        version_source = ast.parse(git("show", f"{source_commit}:lm15/_version.py", cwd=checkout))
        version = next(
            ast.literal_eval(node.value)
            for node in version_source.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        )
        if not isinstance(version, str) or not version or "\n" in version:
            parser.error("Source package must declare a nonempty, single-line version string.")
        license_text = git("show", f"{source_commit}:LICENSE", cwd=checkout)
        git("cat-file", "-e", f"{source_commit}:lm15/__init__.py", cwd=checkout)
        split = git("subtree", "split", "--prefix=lm15", source_commit, cwd=checkout)
        git("fetch", str(checkout), split)

        initialized = RECORD.exists()
        if initialized:
            fields = dict(
                line.split("=", 1) for line in RECORD.read_text().splitlines() if "=" in line
            )
            # A GitHub squash can remove every subtree parent and trailer.
            # Recover the old package from its source pin, not DSPy's history.
            for key in ("commit", "split"):
                value = fields.get(key, "")
                if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
                    parser.error(f"Invalid {key} in {RECORD.name}; refusing to update.")
            git("fetch", "origin", fields["commit"], cwd=checkout)
            old_split = git("subtree", "split", "--prefix=lm15", fields["commit"], cwd=checkout)
            if old_split != fields["split"]:
                parser.error("Recorded split does not match the pinned source; refusing to update.")
            git("fetch", str(checkout), old_split)
            old_tree = git("rev-parse", f"{old_split}^{{tree}}")
            if git("rev-parse", f"HEAD:{PREFIX}") != old_tree:
                parser.error("Vendored files differ from the recorded source; move edits upstream first.")
            if (fields.get("split"), fields.get("commit"), fields.get("version"), fields.get("contract")) == (
                split, source_commit, version, contract
            ):
                print("lm15 is already at this source commit.")
                return 0
        elif (REPO_ROOT / PREFIX).exists():
            if git("ls-files", "--", PREFIX):
                if not (REPO_ROOT / PREFIX / "VENDORED").exists():
                    parser.error("Existing package has no legacy VENDORED record; refusing to replace it.")
                git("rm", "-r", PREFIX)
                git("commit", "-m", "Remove copied lm15 before establishing its subtree")
            # Git leaves ignored files (such as bytecode) behind. Preserve them
            # outside the prefix, which subtree add requires to be absent.
            if (REPO_ROOT / PREFIX).exists():
                backup = Path(tempfile.mkdtemp(prefix="dspy-lm15-leftovers-")) / "lm15"
                (REPO_ROOT / PREFIX).rename(backup)
                print(f"Preserved untracked package leftovers at {backup}")

        message = (
            f"{'Update' if initialized else 'Vendor'} lm15 package subtree\n\n"
            f"Source: {args.source}\nPython commit: {source_commit}\n"
            f"Contract: {contract}\nPackage split: {split}"
        )
        try:
            if initialized:
                # Reconstruct a genuine package-root squash parent. A metadata-
                # only commit with DSPy's root tree would be the wrong merge base.
                anchor = git(
                    "commit-tree", old_tree, "-m",
                    f"Restore lm15 subtree base from source pin\n\n"
                    f"git-subtree-dir: {PREFIX}\ngit-subtree-split: {old_split}",
                )
                mainline = git("rev-parse", "HEAD")
                git(
                    "merge", "--strategy=ours", "--no-ff", "--allow-unrelated-histories",
                    "-m", f"Reconnect lm15 subtree history without changing files\n\n"
                    f"git-subtree-dir: {PREFIX}\ngit-subtree-mainline: {mainline}\n"
                    f"git-subtree-split: {old_split}", anchor,
                )
            git("subtree", "merge" if initialized else "add", f"--prefix={PREFIX}",
                "--squash", "-m", message, split)
            RECORD.write_text(
                f"source={args.source}\nversion={version}\ncommit={source_commit}\ncontract={contract}\nsplit={split}\n"
            )
            LICENSE.write_text(license_text + "\n")
            git("add", str(RECORD), str(LICENSE))
            if git("diff", "--cached", "--name-only"):
                git("commit", "-m", f"Record lm15 source and contract for {source_commit[:12]}")
        except subprocess.CalledProcessError:
            print(f"Update stopped. Inspect git status before continuing. Previous HEAD: {before}")
            raise
    print(f"Imported lm15 {source_commit} (contract {contract}). Local commits only; review before pushing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
