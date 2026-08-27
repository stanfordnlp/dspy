#!/usr/bin/env python3
"""Build DSPy documentation with Zensical from a disposable prepared project."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import yaml

UNSUPPORTED_PLUGINS = {"social", "mkdocs-jupyter", "redirects", "llmstxt"}


class TaggedScalar(str):
    """A YAML scalar whose application-specific tag must survive a round trip."""

    def __new__(cls, value: str, tag: str):
        scalar = super().__new__(cls, value)
        scalar.tag = tag
        return scalar


class ConfigLoader(yaml.SafeLoader):
    pass


class ConfigDumper(yaml.SafeDumper):
    pass


def _load_python_name(loader: ConfigLoader, suffix: str, node: yaml.Node) -> TaggedScalar:
    return TaggedScalar(loader.construct_scalar(node), f"tag:yaml.org,2002:python/name:{suffix}")


def _dump_tagged_scalar(dumper: ConfigDumper, value: TaggedScalar) -> yaml.Node:
    return dumper.represent_scalar(value.tag, str(value))


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _load_python_name)
ConfigDumper.add_representer(TaggedScalar, _dump_tagged_scalar)


def load_config(path: Path) -> dict[str, object]:
    return dict(yaml.load(path.read_text(), Loader=ConfigLoader))


def write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(yaml.dump(config, Dumper=ConfigDumper, sort_keys=False, allow_unicode=True))


def plugin_name(plugin: object) -> str:
    return plugin if isinstance(plugin, str) else str(next(iter(plugin)))


def plugin_settings(config: dict[str, object], name: str) -> dict[str, object]:
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and name in plugin:
            return dict(plugin[name] or {})
    return {}


def replace_notebook_paths(value: object) -> object:
    if isinstance(value, str):
        return value.removesuffix(".ipynb") + ".md" if value.endswith(".ipynb") else value
    if isinstance(value, list):
        return [replace_notebook_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_notebook_paths(item) for key, item in value.items()}
    return value


def nav_entries(value: object) -> dict[str, str]:
    entries: dict[str, str] = {}
    if isinstance(value, list):
        for item in value:
            entries.update(nav_entries(item))
    elif isinstance(value, dict):
        for title, item in value.items():
            if isinstance(item, str) and item.endswith((".md", ".ipynb")):
                entries[item] = str(title)
            else:
                entries.update(nav_entries(item))
    return entries


def route_for_source(path: str) -> str:
    source = Path(path)
    route = (
        source.parent.as_posix().strip("/")
        if source.name == "index.md"
        else source.with_suffix("").as_posix().strip("/")
    )
    return f"/{route}/" if route and route != "." else "/"


def fetch_stats(project: Path) -> dict[str, object]:
    hook = project / "hooks" / "fetch_stats.py"
    spec = importlib.util.spec_from_file_location("dspy_docs_fetch_stats", hook)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {hook}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.fetch_stats())


def prepare_config(
    config: dict[str, object],
    repository: Path,
    stats: dict[str, object],
    *,
    introspect_installed_package: bool = False,
) -> dict[str, object]:
    prepared = dict(config)
    prepared.pop("hooks", None)
    prepared["site_dir"] = "site"
    prepared["nav"] = replace_notebook_paths(prepared.get("nav", []))

    theme = dict(prepared.get("theme", {}))
    theme.pop("name", None)
    theme["variant"] = "classic"
    prepared["theme"] = theme

    extra = dict(prepared.get("extra", {}))
    extra["stats"] = stats
    extra["version"] = {"provider": "mike", "alias": True}
    prepared["extra"] = extra

    extensions = list(prepared.get("markdown_extensions", []))
    if "md_in_html" not in extensions:
        extensions.insert(0, "md_in_html")
    prepared["markdown_extensions"] = extensions

    plugins = []
    for plugin in prepared.get("plugins", []):
        name = plugin_name(plugin)
        if name in UNSUPPORTED_PLUGINS:
            continue
        if name == "mkdocstrings" and isinstance(plugin, dict):
            plugin = dict(plugin)
            settings = dict(plugin[name])
            handlers = dict(settings.get("handlers", {}))
            python = dict(handlers.get("python", {}))
            if introspect_installed_package:
                python.pop("paths", None)
            else:
                python["paths"] = [str(repository)]
            handlers["python"] = python
            settings["handlers"] = handlers
            plugin[name] = settings
        plugins.append(plugin)
    prepared["plugins"] = plugins
    return prepared


def convert_notebooks(docs: Path) -> set[str]:
    from nbconvert import MarkdownExporter

    exporter = MarkdownExporter()
    routes = set()
    for notebook in docs.rglob("*.ipynb"):
        relative = notebook.relative_to(docs).with_suffix(".md").as_posix()
        routes.add(route_for_source(relative).lstrip("/") + "index.html")
        body, resources = exporter.from_filename(
            str(notebook), resources={"output_files_dir": f"{notebook.stem}_files"}
        )
        notebook.with_suffix(".md").write_text(body.replace("<details>", '<details markdown="1">'))
        for name, content in resources.get("outputs", {}).items():
            output = notebook.parent / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
        notebook.unlink()
    for source in docs.rglob("*.py"):
        relative = source.relative_to(docs).with_suffix(".md").as_posix()
        routes.add(route_for_source(relative).lstrip("/") + "index.html")
        source.with_suffix(".md").write_text(
            f"# {source.stem.replace('_', ' ').title()}\n\n```python\n{source.read_text()}\n```\n"
        )
    return routes


def add_missing_titles(docs: Path, entries: dict[str, str]) -> None:
    for source, title in entries.items():
        path = docs / source.replace(".ipynb", ".md")
        if path.name == "index.md" and path == docs / "index.md":
            continue
        if path.exists() and not re.search(r"(?m)^#\s+", path.read_text()):
            path.write_text(f"# {title}\n\n{path.read_text()}")


def redirect_document(target: str) -> str:
    encoded = json.dumps(target)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<link rel="canonical" href="{escape(target, quote=True)}">'
        f"<script>location.replace({encoded}+location.search+location.hash)</script>"
        f'<noscript><meta http-equiv="refresh" content="0; url={escape(target, quote=True)}"></noscript>'
        f'</head><body><a href="{escape(target, quote=True)}">Continue to the documentation</a></body></html>'
    )


def write_redirects(site: Path, redirects: dict[str, str]) -> None:
    for source, target in redirects.items():
        route = route_for_source(source)
        destination = site / route.lstrip("/") / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        target_route = route_for_source(target.replace(".ipynb", ".md"))
        relative = os.path.relpath(target_route.lstrip("/") or ".", route.lstrip("/") or ".")
        if relative == ".":
            relative = "../" if route != "/" else "./"
        elif not relative.endswith("/"):
            relative += "/"
        destination.write_text(redirect_document(relative))


def source_title(source: Path) -> str:
    if source.suffix == ".ipynb":
        notebook = json.loads(source.read_text())
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    else:
        text = source.read_text(errors="replace")
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return re.sub(r"\s*\{[^}]+\}\s*$", "", match.group(1)) if match else source.stem.replace("-", " ").title()


def matching_sources(docs: Path, pattern: str) -> list[Path]:
    if "**." in pattern:
        prefix, suffix = pattern.split("**.", 1)
        return sorted((docs / prefix).rglob(f"*.{suffix}"))
    path = docs / pattern
    return [path] if path.exists() else []


def generate_llms(docs: Path, output: Path, site_url: str, settings: dict[str, object]) -> None:
    parts = [
        "# DSPy",
        "",
        "> DSPy is the framework for programming—rather than prompting—language models.",
        "",
        str(settings["markdown_description"]).strip(),
        "",
    ]
    for section, entries in settings["sections"].items():
        parts.extend((f"## {section}", ""))
        for entry in entries:
            pattern, description = next(iter(entry.items())) if isinstance(entry, dict) else (entry, None)
            for source in matching_sources(docs, pattern):
                route = route_for_source(source.relative_to(docs).as_posix().replace(".ipynb", ".md"))
                target = "index.md" if route == "/" else f"{route.lstrip('/')}index.md"
                line = f"- [{source_title(source)}]({urljoin(site_url, target)})"
                parts.append(f"{line}: {description}" if description else line)
        parts.append("")
    output.write_text("\n".join(parts))


class HeadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metadata: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link" and values.get("rel") == "canonical" and values.get("href"):
            self.metadata["canonical"] = values["href"]
        elif tag == "meta":
            key = values.get("property") or values.get("name")
            if key and values.get("content"):
                self.metadata[key] = values["content"]


def navigation_titles(config: dict[str, object]) -> dict[str, str]:
    titles = {"/": str(config.get("site_name", "DSPy"))}
    for source, title in nav_entries(config.get("nav", [])).items():
        route = route_for_source(source.replace(".ipynb", ".md"))
        if route != "/":
            titles[route] = title
    return titles


def page_titles(config: dict[str, object], docs: Path) -> dict[str, str]:
    titles = navigation_titles(config)
    sources = [*docs.rglob("*.md"), *docs.rglob("*.ipynb"), *docs.rglob("*.py")]
    for source in sources:
        relative = source.relative_to(docs).as_posix()
        route = route_for_source(relative.replace(".ipynb", ".md").replace(".py", ".md"))
        titles.setdefault(route, source_title(source))
    return titles


def write_sitemap(site: Path, site_url: str, docs: Path) -> None:
    sources = [*docs.rglob("*.md"), *docs.rglob("*.ipynb"), *docs.rglob("*.py")]
    locations = {
        urljoin(
            site_url,
            route_for_source(source.relative_to(docs).as_posix().replace(".ipynb", ".md").replace(".py", ".md")).lstrip(
                "/"
            ),
        )
        for source in sources
    }
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{escape(location)}</loc></url>\n" for location in sorted(locations))
        + "</urlset>\n"
    )
    sitemap = site / "sitemap.xml"
    sitemap.write_text(body)
    with gzip.open(site / "sitemap.xml.gz", "wb") as compressed:
        compressed.write(body.encode())


def social_cards(site: Path, site_url: str, titles: dict[str, str], logo: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    card_dir = site / "assets" / "images" / "social-zensical"
    card_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 32)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 84)
    description_font = ImageFont.truetype("DejaVuSans.ttf", 25)
    logo_image = Image.open(logo).convert("RGBA").resize((140, 140))
    for page in site.rglob("*.html"):
        html = page.read_text()
        if "<article" not in html:
            continue
        relative = page.relative_to(site)
        route = "/" if relative == Path("index.html") else f"/{relative.parent.as_posix()}/"
        parser = HeadParser()
        parser.feed(html)
        title = titles.get(route, Path(route.strip("/")).name.replace("-", " ").title() or "DSPy")
        description = parser.metadata.get(
            "description", "The framework for programming—rather than prompting—language models."
        )
        card = card_dir / f"{hashlib.sha256(route.encode()).hexdigest()[:16]}.png"
        image = Image.new("RGB", (1200, 630), "white")
        draw = ImageDraw.Draw(image)
        draw.text((64, 65), "DSPy", fill="black", font=font)
        lines = textwrap.wrap(title, width=23)
        draw.multiline_text((64, 270 if len(lines) > 1 else 170), "\n".join(lines), fill="black", font=title_font)
        image.paste(logo_image, (995, 65), logo_image)
        draw.multiline_text(
            (64, 515), "\n".join(textwrap.wrap(description, width=48)), fill="black", font=description_font, spacing=8
        )
        image.save(card)

        card_url = urljoin(site_url, f"assets/images/social-zensical/{card.name}")
        page_url = urljoin(site_url, route.lstrip("/"))
        metadata = {
            "og:type": "website",
            "og:title": title if route == "/" else f"{title} - DSPy",
            "og:description": description,
            "og:url": page_url,
            "og:image": card_url,
            "og:image:type": "image/png",
            "og:image:width": "1200",
            "og:image:height": "630",
            "twitter:card": "summary_large_image",
            "twitter:title": title if route == "/" else f"{title} - DSPy",
            "twitter:description": description,
            "twitter:image": card_url,
        }
        tags = "".join(
            f'<meta property="{escape(key)}" content="{escape(value, quote=True)}">\n'
            for key, value in metadata.items()
            if key not in parser.metadata
        )
        page.write_text(html.replace("</head>", f"{tags}</head>", 1))


def build_zensical_site(
    *,
    config: Path,
    output: Path,
    python: Path = Path(sys.executable),
    stats: dict[str, object] | None = None,
    keep_project: Path | None = None,
    introspect_installed_package: bool = False,
) -> set[str]:
    source_project = config.parent
    repository = source_project.parent
    source_config = load_config(config)
    redirects = plugin_settings(source_config, "redirects").get("redirect_maps", {})
    llms = plugin_settings(source_config, "llmstxt")
    source_docs = source_project / str(source_config.get("docs_dir", "docs"))
    titles = page_titles(source_config, source_docs)
    stats = stats or fetch_stats(source_project)

    temporary = None if keep_project else tempfile.TemporaryDirectory(prefix="dspy-zensical-build-")
    project = keep_project.resolve() if keep_project else Path(temporary.name) / "project"
    shutil.rmtree(project, ignore_errors=True)
    shutil.copytree(
        source_project, project, ignore=shutil.ignore_patterns("site", ".cache", "__pycache__", ".zensical-*")
    )
    try:
        tabs_override = project / "overrides" / "partials" / "tabs.html"
        tabs_override.unlink(missing_ok=True)
        prepared = prepare_config(
            source_config,
            repository,
            stats,
            introspect_installed_package=introspect_installed_package,
        )
        prepared_config = project / "zensical.generated.yml"
        write_config(prepared_config, prepared)
        notebook_routes = convert_notebooks(project / str(prepared.get("docs_dir", "docs")))
        add_missing_titles(project / str(prepared.get("docs_dir", "docs")), nav_entries(source_config.get("nav", [])))

        subprocess.run(
            [str(python), "-m", "zensical", "build", "--clean", "--config-file", prepared_config],
            cwd=project,
            check=True,
        )
        built = project / str(prepared["site_dir"])
        shutil.rmtree(output, ignore_errors=True)
        shutil.copytree(built, output)
        write_redirects(output, dict(redirects))
        generate_llms(
            source_project / str(source_config.get("docs_dir", "docs")),
            output / "llms.txt",
            str(source_config["site_url"]),
            llms,
        )
        social_cards(output, str(source_config["site_url"]), titles, project / "docs" / "static" / "img" / "logo.png")
        write_sitemap(output, str(source_config["site_url"]), source_docs)
        home = output / "index.html"
        home.write_text(re.sub(r"<title>Index\s+-\s+DSPy</title>", "<title>DSPy</title>", home.read_text(), count=1))
        return notebook_routes
    finally:
        if temporary:
            temporary.cleanup()
