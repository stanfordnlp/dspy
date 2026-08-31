#!/usr/bin/env python3
"""Publish a prebuilt documentation site through Zensical's Mike version store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from html import escape
from pathlib import Path

STABLE_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
DEFAULT_BRANCH = "versioned-docs"
LEGACY_REDIRECTS_MANIFEST = ".dspy-legacy-redirects.json"
HOST_CONFIG = (
    json.dumps(
        {
            "framework": None,
            "trailingSlash": True,
            "headers": [
                {
                    "source": "/(.*).md",
                    "headers": [{"key": "Content-Type", "value": "text/markdown; charset=utf-8"}],
                }
            ],
        },
        indent=2,
    )
    + "\n"
)


def version_tuple(version: str) -> tuple[int, int, int]:
    match = STABLE_VERSION.fullmatch(version)
    if not match or int(match.group(1)) < 3:
        raise argparse.ArgumentTypeError("expected a stable X.Y.Z version at or after 3.0.0")
    return tuple(map(int, match.groups()))


@contextmanager
def working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def deployed_tree_digest(repository: Path, branch: str, identifier: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "ls-tree", "-r", "-z", branch, "--", identifier],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    digest = hashlib.sha256()
    for record in result.stdout.rstrip(b"\0").split(b"\0"):
        metadata, path = record.split(b"\t", 1)
        object_id = metadata.split()[2]
        relative = path.decode().removeprefix(f"{identifier}/")
        digest.update(relative.encode())
        digest.update(subprocess.check_output(["git", "-C", str(repository), "cat-file", "blob", object_id.decode()]))
    return digest.hexdigest()


def branch_file(repository: Path, branch: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{branch}:{path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def redirect_document(target: str) -> str:
    encoded = json.dumps(target)
    escaped = escape(target, quote=True)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<link rel="canonical" href="{escaped}">'
        f"<script>location.replace({encoded}+location.search+location.hash)</script>"
        f'<noscript><meta http-equiv="refresh" content="0; url={escaped}"></noscript>'
        f'</head><body><a href="{escaped}">Continue to the documentation</a></body></html>'
    )


def legacy_redirects(site: Path, identifier: str) -> dict[str, str]:
    redirects = {}
    for page in sorted(site.rglob("*.html")):
        relative = page.relative_to(site)
        if relative == Path("index.html"):
            continue
        if relative.name == "index.html":
            route = relative.parent.as_posix().strip("/")
            if not route:
                continue
            destination = f"/{identifier}/{route}/"
        else:
            route = relative.as_posix()
            destination = f"/{identifier}/{route}"
        top_level = route.split("/", 1)[0]
        if top_level == "current" or re.fullmatch(r"\d+\.\d+(?:\.\d+)?", top_level):
            continue
        redirects[relative.as_posix()] = redirect_document(destination)
    return redirects


def write_legacy_redirects(repository: Path, branch: str, site: Path, identifier: str) -> None:
    from mike import git_utils

    redirects = legacy_redirects(site, identifier)
    previous_text = branch_file(repository, branch, LEGACY_REDIRECTS_MANIFEST)
    previous = set(json.loads(previous_text)) if previous_text else set()
    manifest = json.dumps(sorted(redirects), indent=2) + "\n"
    if previous_text == manifest:
        return
    with git_utils.Commit(branch, "Update unversioned documentation redirects") as commit:
        commit.delete_files(sorted(previous - redirects.keys()))
        for path, document in redirects.items():
            commit.add_file(git_utils.FileInfo(path, document))
        commit.add_file(git_utils.FileInfo(LEGACY_REDIRECTS_MANIFEST, manifest))


def publish_site(
    *,
    repository: Path,
    site: Path,
    identifier: str,
    title: str,
    aliases: list[str],
    renderer: str,
    package_source: str,
    branch: str = DEFAULT_BRANCH,
    mutable: bool = False,
    default: bool = False,
) -> bool:
    from mike import commands, git_utils

    if not mutable:
        # Automated publication is append-only. Intentional corrections to an
        # existing snapshot go through review in the deployment repository.
        version_tuple(identifier)
        deployed = deployed_tree_digest(repository, branch, identifier)
        if deployed:
            if deployed != tree_digest(site):
                raise RuntimeError(f"immutable Mike snapshot {identifier} already exists with different content")
            return False

    config = {"site_dir": str(site), "use_directory_urls": True}
    message = f"Publish DSPy documentation {identifier}"
    with working_directory(repository):
        with commands.deploy(
            config,
            identifier,
            title,
            aliases,
            update_aliases=True,
            alias_type=commands.AliasType.redirect,
            branch=branch,
            message=message,
            set_props=[("renderer", renderer), ("package_source", package_source)],
        ):
            pass
        root = branch_file(repository, branch, "index.html")
        if default and (root is None or f"url={identifier}/" not in root):
            commands.set_default(
                identifier,
                branch=branch,
                message=f"Set default documentation to {identifier}",
            )
        if default:
            write_legacy_redirects(repository, branch, site, identifier)
        if branch_file(repository, branch, "vercel.json") != HOST_CONFIG:
            with git_utils.Commit(branch, "Configure static documentation hosting") as commit:
                commit.add_file(git_utils.FileInfo("vercel.json", HOST_CONFIG))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--title")
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--renderer", choices=("material", "zensical"), required=True)
    parser.add_argument(
        "--package-source",
        choices=("pypi-wheel", "workflow-wheel", "tag-built-wheel", "working-tree"),
        required=True,
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--mutable", action="store_true")
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    changed = publish_site(
        repository=args.repository.resolve(),
        site=args.site.resolve(),
        identifier=args.identifier,
        title=args.title or args.identifier,
        aliases=args.alias,
        renderer=args.renderer,
        package_source=args.package_source,
        branch=args.branch,
        mutable=args.mutable,
        default=args.default,
    )
    print("published" if changed else "already published")


if __name__ == "__main__":
    main()
