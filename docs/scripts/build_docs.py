#!/usr/bin/env python3
"""Build Current or one immutable documentation release snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__:
    from .zensical_build import build_zensical_site
else:  # Direct script execution from the deployed docs repository.
    from zensical_build import build_zensical_site


STABLE_VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
SOURCE_MAP_COMMENT = re.compile(rb"(?:\n?//# sourceMappingURL=[^\r\n]*|/\*# sourceMappingURL=.*?\*/)", re.DOTALL)
ROOT_URL_ATTRIBUTE = re.compile(
    r'(?P<prefix>\b(?:href|src|action)=["\x27])(?P<url>/(?!/)[^"\x27]*)(?P<suffix>["\x27])',
    re.IGNORECASE,
)
VERSIONED_PATH = re.compile(r"^/(?:current|\d+\.\d+(?:\.\d+)?)(?:/|$)")
SHARED_HEADER_STYLES = Path(__file__).parent.parent / "versioning" / "header.css"


def stable_version(value: str) -> str:
    match = STABLE_VERSION.fullmatch(value)
    if not match or int(match.group("major")) < 3:
        raise argparse.ArgumentTypeError("expected a stable X.Y.Z version at or after 3.0.0")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


def patched_config(config: Path, identifier: str, *, edit_ref: str | None = None) -> Path:
    text = config.read_text()
    text, site_count = re.subn(r"(?m)^site_url:\s*.*$", f"site_url: https://dspy.ai/{identifier}/", text, count=1)
    if site_count != 1:
        raise RuntimeError(f"could not patch site_url in {config}")
    if edit_ref:
        text, edit_count = re.subn(r"(?m)^edit_uri:\s*.*$", f"edit_uri: blob/{edit_ref}/docs/docs/", text, count=1)
        if edit_count != 1:
            raise RuntimeError(f"could not patch edit_uri in {config}")
        # Release API reference must import the installed release artifact, not
        # preferentially resolve the source checkout through mkdocstrings.
        text = re.sub(r'(?m)^[ \t]+paths:\s*\[\s*["\x27]?\.\.["\x27]?\s*\][ \t]*\n', "", text, count=1)
    if not re.search(r"(?m)^\s+provider:\s*mike\s*$", text):
        version_config = "    version:\n        provider: mike\n        alias: true\n"
        text, extra_count = re.subn(r"(?m)^extra:\s*$", f"extra:\n{version_config}", text, count=1)
        if extra_count == 0:
            text = f"{text.rstrip()}\n\nextra:\n{version_config}"

    handle, name = tempfile.mkstemp(prefix=f".mkdocs-{identifier}-", suffix=".yml", dir=config.parent)
    os.close(handle)
    output = Path(name)
    output.write_text(text)
    return output


def optimize_site(site: Path) -> dict[str, int]:
    """Remove non-runtime artifacts and minify HTML without changing page behavior."""
    import htmlmin

    before = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    source_maps = list(site.rglob("*.map"))
    for path in source_maps:
        path.unlink()
    for pattern in ("*.js", "*.css"):
        for path in site.rglob(pattern):
            source = path.read_bytes()
            optimized = SOURCE_MAP_COMMENT.sub(b"", source)
            if optimized != source:
                path.write_bytes(optimized)
    for path in site.rglob("*.html"):
        path.write_text(
            htmlmin.minify(
                path.read_text(),
                remove_empty_space=True,
                reduce_empty_attributes=False,
                remove_optional_attribute_quotes=False,
                convert_charrefs=False,
                pre_tags=("pre", "textarea", "code"),
            )
        )
    after = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    return {"before": before, "after": after, "source_maps": len(source_maps)}


def install_shared_header_styles(site: Path, source: Path = SHARED_HEADER_STYLES) -> None:
    """Give every renderer and historical snapshot the same header controls."""
    destination = site / "_static" / "dspy-header.css"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    for page in site.rglob("*.html"):
        relative = Path(os.path.relpath(destination, page.parent)).as_posix()
        tag = f'<link rel="stylesheet" href="{relative}">'
        html = page.read_text()
        if tag not in html:
            page.write_text(html.replace("</head>", f"{tag}</head>", 1))


def scope_root_relative_urls(site: Path, identifier: str) -> None:
    """Keep hand-authored root-relative links inside the selected version."""

    def replace(match: re.Match[str]) -> str:
        url = match.group("url")
        if VERSIONED_PATH.match(url):
            return match.group(0)
        scoped = f"/{identifier}{url}"
        return f"{match.group('prefix')}{scoped}{match.group('suffix')}"

    for page in site.rglob("*.html"):
        html = page.read_text()
        scoped = ROOT_URL_ATTRIBUTE.sub(replace, html)
        if scoped != html:
            page.write_text(scoped)


def installed_packages() -> dict[str, str]:
    return dict(
        sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        )
    )


def renderer_version(renderer: str) -> str:
    package = "mkdocs-material" if renderer == "material" else "zensical"
    return importlib.metadata.version(package)


def validate_release_site(site: Path, config: Path, version: str, renderer: str) -> None:
    search_index = site / "search" / "search_index.json" if renderer == "material" else site / "search.json"
    expected = (site / "index.html", site / "api" / "index.html", search_index)
    missing = [str(path.relative_to(site)) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"release site is missing required output: {', '.join(missing)}")

    home = (site / "index.html").read_text()
    canonical = f'<link rel="canonical" href="https://dspy.ai/{version}/">'
    if canonical not in home:
        raise RuntimeError(f"release home page is missing canonical URL {canonical}")

    docs_dir = config.parent / "docs"
    missing_notebooks = []
    for notebook in docs_dir.rglob("*.ipynb"):
        relative = notebook.relative_to(docs_dir)
        outputs = (
            site / relative.with_suffix(".html"),
            site / relative.with_suffix("") / "index.html",
        )
        if not any(output.exists() for output in outputs):
            missing_notebooks.append(str(notebook.relative_to(docs_dir)))
    if missing_notebooks:
        raise RuntimeError(f"notebooks were not rendered: {', '.join(missing_notebooks)}")

    config_text = config.read_text()
    if re.search(r"(?m)^\s*- social\s*$", config_text):
        cards = site / "assets" / "images" / ("social" if renderer == "material" else "social-zensical")
        if not cards.exists() or not any(cards.rglob("*.png")) or 'property="og:image"' not in home:
            raise RuntimeError("social cards or Open Graph metadata were not generated")
    if re.search(r"(?m)^\s*- llmstxt:\s*$", config_text) and not (site / "llms.txt").exists():
        raise RuntimeError("mkdocs-llmstxt was configured but llms.txt was not generated")

    try:
        dspy_version = importlib.metadata.version("dspy")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("release build environment does not contain DSPy") from error
    if dspy_version != version:
        raise RuntimeError(f"release build imported dspy=={dspy_version}, expected {version}")


def build(
    *,
    config: Path,
    output: Path,
    renderer: str,
    version: str | None = None,
    artifact: Path | None = None,
    package_source: str | None = None,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    identifier = version or "current"
    effective_config = patched_config(config, identifier, edit_ref=version)
    try:
        if renderer == "material":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mkdocs",
                    "build",
                    "--clean",
                    "--config-file",
                    str(effective_config),
                    "--site-dir",
                    str(output),
                ],
                cwd=config.parent,
                check=True,
            )
        else:
            build_zensical_site(
                config=effective_config,
                output=output,
                python=Path(sys.executable),
                introspect_installed_package=version is not None,
            )
    finally:
        effective_config.unlink(missing_ok=True)

    if version:
        if artifact is None or package_source is None:
            raise ValueError("release builds require an artifact and package source")
    scope_root_relative_urls(output, identifier)
    if version:
        validate_release_site(output, config, version, renderer)
    install_shared_header_styles(output)
    optimization = optimize_site(output)
    if version:
        repository = config.parent.parent
        source_commit = git_value(repository, "rev-list", "-n", "1", version)
        metadata = {
            "version": version,
            "source_tag": version,
            "source_commit": source_commit,
            "source_commit_time": git_value(repository, "show", "-s", "--format=%cI", source_commit),
            "renderer": renderer,
            "renderer_version": renderer_version(renderer),
            "package_source": package_source,
            "package_artifact": artifact.name,
            "package_sha256": sha256(artifact),
            "optimization": optimization,
            "python": sys.version.split()[0],
            "packages": installed_packages(),
            "intentional_differences": [
                "The renderer's Mike version selector is enabled in a transient build configuration.",
                "Build provenance metadata added under _meta/build.json.",
                "Generated HTML is minified and source maps are omitted from production snapshots.",
                "Original deployment dependencies were not locked; non-DSPy dependencies are reconstructed as of the tag date.",
            ],
        }
        metadata_dir = output / "_meta"
        metadata_dir.mkdir(exist_ok=True)
        (metadata_dir / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("current", "release"))
    parser.add_argument("--config", type=Path, default=Path("mkdocs.yml"))
    parser.add_argument("--output", type=Path, default=Path("site"))
    parser.add_argument("--renderer", choices=("material", "zensical"), default="material")
    parser.add_argument("--version", type=stable_version)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--package-source", choices=("pypi-wheel", "workflow-wheel", "tag-built-wheel"))
    args = parser.parse_args()
    if args.mode == "release" and not (args.version and args.artifact and args.package_source):
        parser.error("release requires --version, --artifact, and --package-source")
    return args


def main() -> None:
    args = parse_args()
    build(
        config=args.config.resolve(),
        output=args.output,
        renderer=args.renderer,
        version=args.version if args.mode == "release" else None,
        artifact=args.artifact.resolve() if args.artifact else None,
        package_source=args.package_source,
    )


if __name__ == "__main__":
    main()
