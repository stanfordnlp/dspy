import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from docs.scripts.build_docs import (
    install_shared_header,
    patched_config,
    release_version,
    remove_source_maps,
    scope_root_relative_urls,
    set_release_badge_version,
    validate_release_site,
)
from docs.scripts.publish_versioned_docs import publish_site, require_current_renderer, version_tuple

requires_mike = pytest.mark.skipif(importlib.util.find_spec("mike") is None, reason="Mike is a docs-only dependency")


def test_current_workflow_publishes_to_production_branch():
    workflow = Path(".github/workflows/docs-push.yml").read_text()
    checkout = workflow.split("      - name: Check out documentation deployment", 1)[1].split("      - name:", 1)[0]
    publish = workflow.split("      - name: Publish Current through Mike", 1)[1]

    assert "ref: master" in checkout
    assert "--branch master" in publish
    assert "push origin master" in publish
    assert "versioned-docs" not in checkout + publish


def make_site(root, content: str):
    site = root / "site"
    nested = site / "guide"
    nested.mkdir(parents=True)
    (site / "index.html").write_text(f"<html><head></head><body>{content}</body></html>")
    (nested / "index.html").write_text("<html><head></head><body>Guide</body></html>")
    return site


def make_repository(root):
    repository = root / "deployment"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
    (repository / "README").write_text("deployment\n")
    subprocess.run(["git", "add", "README"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "Initial deployment"], cwd=repository, check=True)
    return repository


def branch_file(repository, branch: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{branch}:{path}"], cwd=repository, text=True)


def set_current_renderer(repository, renderer: str):
    inventory = json.loads(branch_file(repository, "versioned-docs", "versions.json"))
    current = next(entry for entry in inventory if entry["version"] == "current")
    current["properties"]["renderer"] = renderer
    subprocess.run(["git", "checkout", "-q", "versioned-docs"], cwd=repository, check=True)
    (repository / "versions.json").write_text(json.dumps(inventory, indent=2) + "\n")
    subprocess.run(["git", "add", "versions.json"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", f"Set Current renderer to {renderer}"], cwd=repository, check=True)
    subprocess.run(["git", "checkout", "-q", "master"], cwd=repository, check=True)


def test_release_versions_accept_prereleases_and_reject_old_major_versions():
    assert release_version("3.1.2") == "3.1.2"
    assert release_version("3.4.0b1") == "3.4.0b1"
    assert version_tuple("3.4.0b1") < version_tuple("3.4.0")
    with pytest.raises(argparse.ArgumentTypeError):
        release_version("3.4.0.post1")
    with pytest.raises(argparse.ArgumentTypeError):
        version_tuple("2.6.0")


def test_release_config_enables_mike_and_scopes_urls(tmp_path):
    config = tmp_path / "mkdocs.yml"
    config.write_text(
        "site_url: https://dspy.ai/\nedit_uri: blob/main/docs/docs/\nsite_name: DSPy\n"
        'plugins:\n    - mkdocstrings:\n        handlers:\n            python:\n                paths: [".."]\n'
        "extra:\n    social: []\n"
    )

    result = patched_config(config, "3.2.1", edit_ref="3.2.1")
    try:
        text = result.read_text()
        assert "site_url: https://dspy.ai/3.2.1/" in text
        assert "edit_uri: blob/3.2.1/docs/docs/" in text
        assert 'paths: [".."]' not in text
        assert "provider: mike" in text
        assert "alias: true" in text
    finally:
        result.unlink()


def test_release_validation_accepts_zensical_minified_metadata(tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "api").mkdir(parents=True)
    (site / "assets" / "images" / "social-zensical").mkdir(parents=True)
    (site / "index.html").write_text(
        "<html><head>"
        "<link rel=canonical href=https://dspy.ai/3.4.0b1/>"
        "<meta property=og:image content=https://dspy.ai/card.png>"
        "</head></html>"
    )
    (site / "api" / "index.html").write_text("API")
    (site / "search.json").write_text("{}")
    (site / "llms.txt").write_text("DSPy")
    (site / "assets" / "images" / "social-zensical" / "card.png").write_bytes(b"png")
    config = tmp_path / "mkdocs.yml"
    config.write_text("site_name: DSPy\n")
    monkeypatch.setattr("docs.scripts.build_docs.importlib.metadata.version", lambda package: "3.4.0b1")

    validate_release_site(site, config, "3.4.0b1")


def test_production_build_removes_source_maps(tmp_path):
    site = tmp_path / "site"
    assets = site / "assets"
    assets.mkdir(parents=True)
    page = site / "index.html"
    page.write_text(
        "<!doctype html>\n<html>\n  <body>\n    <p>Hello world</p>\n"
        "    <pre><code>first\n  second</code></pre>\n  </body>\n</html>\n"
    )
    (assets / "app.js").write_text("value();\n//# sourceMappingURL=app.js.map\n")
    (assets / "app.js.map").write_text("{}\n")

    result = remove_source_maps(site)

    assert "first\n  second" in page.read_text()
    assert result["after"] < result["before"]
    assert result["source_maps"] == 1
    assert not (assets / "app.js.map").exists()
    assert "sourceMappingURL" not in (assets / "app.js").read_text()


def test_shared_header_assets_are_installed_on_nested_pages(tmp_path):
    site = make_site(tmp_path, "Home")

    install_shared_header(site)
    install_shared_header(site)

    assert (site / "_static" / "dspy-header.css").is_file()
    assert (site / "_static" / "dspy-header.js").is_file()
    home = (site / "index.html").read_text()
    nested = (site / "guide" / "index.html").read_text()
    assert home.count('href="_static/dspy-header.css"') == 1
    assert home.count('src="_static/dspy-header.js"') == 1
    assert nested.count('href="../_static/dspy-header.css"') == 1
    assert nested.count('src="../_static/dspy-header.js"') == 1


def test_root_relative_links_stay_inside_the_selected_version(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    page = site / "index.html"
    page.write_text(
        '<a href="/learn/">Learn</a><img src="/assets/logo.svg">'
        '<a href="/3.2.1/api/">Other version</a><a href="/3.4.0b1/api/">Beta</a>'
        '<a href="//example.com/path">External</a>'
    )

    scope_root_relative_urls(site, "3.3.1")

    html = page.read_text()
    assert 'href="/3.3.1/learn/"' in html
    assert 'src="/3.3.1/assets/logo.svg"' in html
    assert 'href="/3.2.1/api/"' in html
    assert 'href="/3.4.0b1/api/"' in html
    assert 'href="//example.com/path"' in html


def test_release_badge_uses_snapshot_version_without_changing_other_content(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    home = site / "index.html"
    home.write_text(
        '<div class="hp-hero-badge">\n<span></span>\nDSPy 3.4.0b1 &mdash; Historical blurb\n</div>'
        '<p>Documentation example for DSPy 3.4.0b1</p>'
    )

    set_release_badge_version(site, "3.3.1")

    html = home.read_text()
    assert "DSPy 3.3.1 &mdash; Historical blurb" in html
    assert "Documentation example for DSPy 3.4.0b1" in html


@requires_mike
def test_mike_preserves_patches_and_moves_minor_redirect(tmp_path):
    repository = make_repository(tmp_path)
    first = make_site(tmp_path / "first", "3.0.0")
    second = make_site(tmp_path / "second", "3.0.1")

    for version, site in (("3.0.0", first), ("3.0.1", second)):
        publish_site(
            repository=repository,
            site=site,
            identifier=version,
            aliases=["3.0"],
            package_source="pypi-wheel",
        )

    assert "3.0.0" in branch_file(repository, "versioned-docs", "3.0.0/index.html")
    assert "3.0.1" in branch_file(repository, "versioned-docs", "3.0.1/index.html")
    alias = branch_file(repository, "versioned-docs", "3.0/guide/index.html")
    assert "../../3.0.1/guide/" in alias

    inventory = json.loads(branch_file(repository, "versioned-docs", "versions.json"))
    assert [entry["version"] for entry in inventory] == ["3.0.1", "3.0.0"]
    assert inventory[0]["aliases"] == ["3.0"]
    assert inventory[1]["aliases"] == []


@requires_mike
def test_delayed_older_patch_does_not_move_minor_redirect_backward(tmp_path):
    repository = make_repository(tmp_path)
    newest = make_site(tmp_path / "newest", "3.0.1")
    delayed = make_site(tmp_path / "delayed", "3.0.0")

    for version, site in (("3.0.1", newest), ("3.0.0", delayed)):
        publish_site(
            repository=repository,
            site=site,
            identifier=version,
            aliases=["3.0"],
            package_source="workflow-wheel",
        )

    alias = branch_file(repository, "versioned-docs", "3.0/guide/index.html")
    assert "../../3.0.1/guide/" in alias
    inventory = json.loads(branch_file(repository, "versioned-docs", "versions.json"))
    aliases = {entry["version"]: entry["aliases"] for entry in inventory}
    assert aliases == {"3.0.0": [], "3.0.1": ["3.0"]}


@requires_mike
def test_mike_refuses_to_replace_an_immutable_snapshot(tmp_path):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "first", "original")
    arguments = {
        "repository": repository,
        "site": site,
        "identifier": "3.4.0b1",
        "aliases": [],
        "package_source": "pypi-wheel",
    }
    assert publish_site(**arguments)
    assert not publish_site(**arguments)
    (site / "index.html").write_text("different")

    with pytest.raises(RuntimeError, match="immutable Mike snapshot"):
        publish_site(**arguments)


@requires_mike
def test_mike_current_is_mutable_and_default(tmp_path):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "current", "first")
    arguments = {
        "repository": repository,
        "site": site,
        "identifier": "current",
        "aliases": [],
        "package_source": "working-tree",
    }

    publish_site(**arguments)
    (site / "index.html").write_text("second")
    publish_site(**arguments)

    assert branch_file(repository, "versioned-docs", "current/index.html") == "second"
    assert "url=current/" in branch_file(repository, "versioned-docs", "index.html")
    assert 'location.replace("/current/guide/"' in branch_file(repository, "versioned-docs", "guide/index.html")
    host_config = json.loads(branch_file(repository, "versioned-docs", "vercel.json"))
    assert host_config["framework"] is None
    assert host_config["buildCommand"] == "true"
    assert host_config["outputDirectory"] == "."
    assert host_config["redirects"] == [
        {
            "source": r"/:version(\d+\.\d+(?:\.\d+(?:(?:a|b|rc)\d+)?)?)",
            "destination": "/:version/",
            "permanent": True,
        },
        {
            "source": "/:path((?:.*/)?[^./]+)",
            "destination": "/:path/",
            "permanent": True,
        },
    ]
    assert "trailingSlash" not in host_config
    production_inventory = subprocess.run(
        ["git", "cat-file", "-e", "master:versions.json"], cwd=repository, capture_output=True
    )
    assert production_inventory.returncode != 0
    require_current_renderer(repository, "versioned-docs", "zensical")
    with pytest.raises(RuntimeError, match="expected 'material'"):
        require_current_renderer(repository, "versioned-docs", "material")


@requires_mike
def test_publication_requires_reviewed_zensical_current(tmp_path):
    repository = make_repository(tmp_path)
    deployed = make_site(tmp_path / "deployed", "deployed")
    publish_site(
        repository=repository,
        site=deployed,
        identifier="current",
        aliases=[],
        package_source="working-tree",
    )
    set_current_renderer(repository, "material")

    update = make_site(tmp_path / "update", "update")
    arguments = {
        "repository": repository,
        "site": update,
        "identifier": "current",
        "aliases": [],
        "package_source": "working-tree",
        "required_current_renderer": "zensical",
    }
    with pytest.raises(RuntimeError, match="expected 'zensical'"):
        publish_site(**arguments)
    assert "deployed" in branch_file(repository, "versioned-docs", "current/index.html")

    set_current_renderer(repository, "zensical")
    assert publish_site(**arguments)
    assert "update" in branch_file(repository, "versioned-docs", "current/index.html")


@requires_mike
def test_mike_removes_stale_unversioned_redirects(tmp_path):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "current", "first")
    stale = site / "removed"
    stale.mkdir()
    (stale / "index.html").write_text("Removed")
    arguments = {
        "repository": repository,
        "site": site,
        "identifier": "current",
        "aliases": [],
        "package_source": "working-tree",
    }

    publish_site(**arguments)
    (stale / "index.html").unlink()
    stale.rmdir()
    publish_site(**arguments)

    result = subprocess.run(
        ["git", "cat-file", "-e", "versioned-docs:removed/index.html"],
        cwd=repository,
        capture_output=True,
    )
    assert result.returncode != 0
