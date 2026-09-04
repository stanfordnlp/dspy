import gzip
import json
from pathlib import Path

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
    prepare_config,
    redirect_maps,
    remove_redirect_sources,
    validate_output,
    write_redirects,
)


def test_zensical_configuration_is_native_and_keeps_build_compatibility_separate():
    project = Path(__file__).parents[2] / "docs"
    config = load_config(project / "mkdocs.yml")
    build = load_config(project / "build.yml")

    assert config["theme"]["variant"] == "classic"
    assert "name" not in config["theme"]
    assert [plugin if isinstance(plugin, str) else next(iter(plugin)) for plugin in config["plugins"]] == [
        "search",
        "minify",
        "mkdocstrings",
        "redirects",
    ]
    assert "hooks" not in config
    redirects = next(plugin["redirects"] for plugin in config["plugins"] if "redirects" in plugin)
    assert redirects["redirect_maps"]["intro/index.md"] == "index.md"
    assert "Tutorials - Notebooks" in build["llms"]["sections"]


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


def test_extracts_navigation_titles(tmp_path):
    config = tmp_path / "mkdocs.yml"
    config.write_text(
        "nav:\n    - Overview: index.md\n    - Retrieval-Augmented Generation (RAG): tutorials/rag/index.md\n"
    )

    parsed = load_config(config)

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


def test_notebook_conversion_preserves_python_pages(tmp_path):
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


def test_prepared_config_exposes_stats_without_mutating_source():
    source = {"extra": {"social": []}}

    prepared = prepare_config(source, {"stars": "10k"})

    assert prepared["extra"] == {"social": [], "stats": {"stars": "10k"}}
    assert source == {"extra": {"social": []}}


def test_redirect_sources_are_removed_only_from_prepared_tree(tmp_path):
    source = tmp_path / "learn" / "index.md"
    source.parent.mkdir()
    source.write_text("legacy content")
    config = {"plugins": [{"redirects": {"redirect_maps": {"learn/index.md": "index.md"}}}]}

    remove_redirect_sources(tmp_path, config)

    assert not source.exists()
    assert redirect_maps(config) == {"learn/index.md": "index.md"}


def test_redirects_preserve_relative_navigation_query_and_fragment(tmp_path):
    write_redirects(
        tmp_path,
        {
            "learn/index.md": "getting-started/first-program.md",
            "api/old.md": "api/new/index.md",
        },
    )

    learn = (tmp_path / "learn" / "index.html").read_text()
    api = (tmp_path / "api" / "old" / "index.html").read_text()
    assert 'location.replace("../getting-started/first-program/"+location.search+location.hash)' in learn
    assert 'href="../getting-started/first-program/"' in learn
    assert 'location.replace("../new/"+location.search+location.hash)' in api


def test_output_validation_covers_native_and_compatibility_features(tmp_path):
    required = ("index.html", "api/index.html", "llms.txt", "sitemap.xml")
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<article></article><meta property="og:image">')
    (tmp_path / "search.json").write_text(json.dumps({"items": [{"location": "/"}]}))
    with gzip.open(tmp_path / "sitemap.xml.gz", "wb") as compressed:
        compressed.write((tmp_path / "sitemap.xml").read_bytes())
    notebook = tmp_path / "tutorial" / "index.html"
    notebook.parent.mkdir()
    notebook.write_text("notebook")
    redirect = tmp_path / "old" / "index.html"
    redirect.parent.mkdir()
    redirect.write_text("redirect")
    cards = tmp_path / "assets" / "images" / "social-zensical"
    cards.mkdir(parents=True)
    (cards / "home.png").write_bytes(b"card")

    validate_output(tmp_path, {"tutorial/index.html"}, {"old.md": "index.md"})

    (tmp_path / "search.json").write_text(json.dumps({"items": []}))
    with pytest.raises(RuntimeError, match="search index is empty"):
        validate_output(tmp_path, {"tutorial/index.html"}, {"old.md": "index.md"})
