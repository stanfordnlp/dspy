import gzip
import json

import pytest

from docs.hooks import fetch_stats as stats_hook
from docs.scripts.build_docs import install_shared_header_styles
from docs.scripts.zensical_build import (
    add_missing_titles,
    convert_notebooks,
    generate_llms,
    load_config,
    nav_entries,
    navigation_titles,
    plugin_settings,
    prepare_config,
    route_for_source,
    write_config,
    write_redirects,
    write_sitemap,
)
from docs.scripts.zensical_parity import (
    PageParser,
    redirect_target,
)


def test_zensical_config_removes_only_unsupported_plugins(tmp_path):
    config = tmp_path / "mkdocs.yml"
    config.write_text(
        """theme:
    name: material
plugins:
    - social
    - search:
        lang: en
    - mkdocstrings:
        handlers:
            python:
                paths: [".."]
    - mkdocs-jupyter:
        ignore_h1_titles: true
    - redirects:
        redirect_maps:
            "old.md": "guide.ipynb"
    - llmstxt:
        sections:
            Guides:
                - guide.ipynb
hooks:
    - hooks/fetch_stats.py
extra:
    social: []
markdown_extensions:
    - toc
nav:
    - Guide: guide.ipynb
"""
    )

    result = prepare_config(load_config(config), tmp_path, {"stars": "10k"})

    assert result["theme"] == {"variant": "classic"}
    assert [plugin if isinstance(plugin, str) else next(iter(plugin)) for plugin in result["plugins"]] == [
        "search",
        "mkdocstrings",
    ]
    assert "hooks" not in result
    assert result["plugins"][1]["mkdocstrings"]["handlers"]["python"]["paths"] == [str(tmp_path)]
    assert result["extra"] == {
        "social": [],
        "stats": {"stars": "10k"},
        "version": {"provider": "mike", "alias": True},
    }
    assert "md_in_html" in result["markdown_extensions"]
    assert result["nav"] == [{"Guide": "guide.md"}]


def test_release_config_introspects_installed_package(tmp_path):
    config = {
        "plugins": [{"mkdocstrings": {"handlers": {"python": {"paths": [".."], "options": {"show_source": True}}}}}]
    }

    result = prepare_config(config, tmp_path, {}, introspect_installed_package=True)

    python = result["plugins"][0]["mkdocstrings"]["handlers"]["python"]
    assert "paths" not in python
    assert python["options"] == {"show_source": True}


def test_shared_header_styles_use_page_relative_links(tmp_path):
    site = tmp_path / "site"
    nested = site / "guide" / "example"
    nested.mkdir(parents=True)
    (site / "index.html").write_text("<html><head></head><body>Home</body></html>")
    (nested / "index.html").write_text("<html><head></head><body>Guide</body></html>")
    source = tmp_path / "header.css"
    source.write_text(".md-version { width: 10rem; }\n")

    install_shared_header_styles(site, source)

    assert '<link rel="stylesheet" href="_static/dspy-header.css">' in (site / "index.html").read_text()
    assert '<link rel="stylesheet" href="../../_static/dspy-header.css">' in (nested / "index.html").read_text()
    assert (site / "_static" / "dspy-header.css").read_text() == source.read_text()


def test_extracts_plugin_configuration_and_navigation_titles(tmp_path):
    text = """nav:
    - Overview: index.md
    - Retrieval-Augmented Generation (RAG): tutorials/rag/index.ipynb
plugins:
    - search
    - llmstxt:
        markdown_description: Description
        sections:
            Guides:
                - tutorials/**.ipynb
hooks:
    - hook.py
"""
    config = tmp_path / "mkdocs.yml"
    config.write_text(text)

    parsed = load_config(config)
    llms = plugin_settings(parsed, "llmstxt")

    assert llms["sections"] == {"Guides": ["tutorials/**.ipynb"]}
    assert navigation_titles(parsed) == {
        "/": "DSPy",
        "/tutorials/rag/": "Retrieval-Augmented Generation (RAG)",
    }


def test_adds_navigation_title_only_when_a_page_has_no_heading(tmp_path):
    docs = tmp_path / "docs"
    (docs / "tutorials" / "classification").mkdir(parents=True)
    missing = docs / "tutorials" / "classification" / "index.md"
    missing.write_text("Page content\n")
    existing = docs / "guide.md"
    existing.write_text("# Existing\n")
    config = tmp_path / "mkdocs.yml"
    config.write_text("nav:\n    - Classification: tutorials/classification/index.md\n    - Guide: guide.md\n")

    add_missing_titles(docs, nav_entries(load_config(config)["nav"]))

    assert missing.read_text().startswith("# Classification\n\n")
    assert existing.read_text() == "# Existing\n"


def test_routes_and_redirects_preserve_nested_targets(tmp_path):
    site = tmp_path / "site"
    site.mkdir()

    redirects = {"intro/index.md": "index.md", "old/guide.md": "tutorials/rag/index.ipynb"}
    write_redirects(site, redirects)

    assert route_for_source("index.md") == "/"
    assert route_for_source("guides/index.md") == "/guides/"
    assert route_for_source("guides/setup.md") == "/guides/setup/"
    assert redirect_target(site, "/intro/") == "/"
    assert redirect_target(site, "/old/guide/") == "/tutorials/rag/"


def test_config_round_trip_preserves_material_python_tags(tmp_path):
    config = tmp_path / "mkdocs.yml"
    config.write_text(
        "markdown_extensions:\n"
        "    - pymdownx.emoji:\n"
        "        emoji_index: !!python/name:material.extensions.emoji.twemoji\n"
    )

    generated = tmp_path / "zensical.yml"
    write_config(generated, load_config(config))

    assert "!!python/name:material.extensions.emoji.twemoji" in generated.read_text()
    assert load_config(generated)["markdown_extensions"]


def test_notebook_conversion_preserves_python_pages_supported_by_mkdocs_jupyter(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Guide")])
    nbformat.write(notebook, tmp_path / "guide.ipynb")
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 1\n")

    routes = convert_notebooks(tmp_path)

    assert routes == {"guide/index.html", "helper/index.html"}
    assert (tmp_path / "guide.md").exists()
    assert helper.exists()
    assert "```python\nVALUE = 1" in (tmp_path / "helper.md").read_text()


def test_page_parser_scopes_content_and_collects_metadata():
    parser = PageParser()
    parser.feed(
        """<html><head><link rel="canonical" href="https://dspy.ai/guide/">
<meta property="og:title" content="Guide - DSPy"><style>ignored style text</style></head>
<body>navigation text<article><h1 id="guide">Guide</h1><p>Useful content</p>
<script>ignored script text</script></article>footer text</body></html>"""
    )

    assert parser.metadata == {"canonical": "https://dspy.ai/guide/", "og:title": "Guide - DSPy"}
    assert parser.headings == [("guide", "Guide")]
    assert parser.content_text == ["Guide", "Useful content"]
    assert "navigation text" in parser.text
    assert "ignored script text" not in parser.text


def test_llms_generation_uses_the_configured_inventory(tmp_path):
    docs = tmp_path / "docs"
    (docs / "guides").mkdir(parents=True)
    (docs / "index.md").write_text("# DSPy\n")
    (docs / "guides" / "first.md").write_text("# First Guide\n")
    (docs / "excluded.md").write_text("# Excluded\n")
    output = tmp_path / "llms.txt"

    generate_llms(
        docs,
        output,
        "https://dspy.ai/",
        {
            "markdown_description": "Description",
            "sections": {
                "Home": [{"index.md": "Overview"}],
                "Guides": ["guides/**.md"],
            },
        },
    )

    result = output.read_text()
    assert "Description" in result
    assert "[DSPy](https://dspy.ai/index.md): Overview" in result
    assert "[First Guide](https://dspy.ai/guides/first/index.md)" in result
    assert "Excluded" not in result


def test_sitemap_uses_source_routes_and_writes_matching_gzip(tmp_path):
    docs = tmp_path / "docs"
    (docs / "guide").mkdir(parents=True)
    (docs / "index.md").write_text("# Home\n")
    (docs / "guide" / "index.ipynb").write_text("{}\n")
    (docs / "example.py").write_text("VALUE = 1\n")
    site = tmp_path / "site"
    site.mkdir()

    write_sitemap(site, "https://dspy.ai/", docs)

    sitemap = (site / "sitemap.xml").read_text()
    assert "<loc>https://dspy.ai/</loc>" in sitemap
    assert "<loc>https://dspy.ai/guide/</loc>" in sitemap
    assert "<loc>https://dspy.ai/example/</loc>" in sitemap
    assert gzip.decompress((site / "sitemap.xml.gz").read_bytes()).decode() == sitemap


def test_public_stats_api_does_not_leak_cache_metadata(tmp_path, monkeypatch):
    cache = tmp_path / "stats.json"
    fetched = {"stars": "10k"}
    monkeypatch.setattr(stats_hook, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(stats_hook, "CACHE_FILE", cache)
    monkeypatch.setattr(stats_hook, "_fetch_all", lambda: fetched)

    result = stats_hook.fetch_stats()

    assert result == {"stars": "10k"}
    assert fetched == {"stars": "10k"}
    assert stats_hook.fetch_stats() == {"stars": "10k"}
    assert set(json.loads(cache.read_text())) == {"stars", "_ts", "_cache_version"}
