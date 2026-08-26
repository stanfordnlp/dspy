import argparse
import json
import subprocess

import pytest

from docs.scripts.build_docs import optimize_site, patched_config, stable_version
from docs.scripts.publish_versioned_docs import publish_site, version_tuple


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


def test_stable_versions_reject_prereleases_and_old_major_versions():
    assert stable_version("3.1.2") == "3.1.2"
    assert version_tuple("3.1.2") == (3, 1, 2)
    with pytest.raises(argparse.ArgumentTypeError):
        stable_version("3.1.2rc1")
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


def test_production_optimization_preserves_preformatted_content(tmp_path):
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

    result = optimize_site(site)

    assert "first\n  second" in page.read_text()
    assert page.stat().st_size < result["before"]
    assert result["after"] < result["before"]
    assert result["source_maps"] == 1
    assert not (assets / "app.js.map").exists()
    assert "sourceMappingURL" not in (assets / "app.js").read_text()


def test_mike_preserves_patches_and_moves_minor_redirect(tmp_path):
    repository = make_repository(tmp_path)
    first = make_site(tmp_path / "first", "3.0.0")
    second = make_site(tmp_path / "second", "3.0.1")

    for version, site in (("3.0.0", first), ("3.0.1", second)):
        publish_site(
            repository=repository,
            site=site,
            identifier=version,
            title=version,
            aliases=["3.0"],
            renderer="material",
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


def test_mike_refuses_to_replace_an_immutable_snapshot(tmp_path):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "first", "original")
    arguments = {
        "repository": repository,
        "site": site,
        "identifier": "3.3.1",
        "title": "3.3.1",
        "aliases": ["3.3"],
        "renderer": "material",
        "package_source": "pypi-wheel",
    }
    assert publish_site(**arguments)
    assert not publish_site(**arguments)
    (site / "index.html").write_text("different")

    with pytest.raises(RuntimeError, match="immutable Mike snapshot"):
        publish_site(**arguments)


def test_mike_current_is_mutable_and_default(tmp_path):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "current", "first")
    arguments = {
        "repository": repository,
        "site": site,
        "identifier": "current",
        "title": "Current",
        "aliases": [],
        "renderer": "zensical",
        "package_source": "working-tree",
        "mutable": True,
        "default": True,
    }

    publish_site(**arguments)
    (site / "index.html").write_text("second")
    publish_site(**arguments)

    assert branch_file(repository, "versioned-docs", "current/index.html") == "second"
    assert "url=current/" in branch_file(repository, "versioned-docs", "index.html")
    assert json.loads(branch_file(repository, "versioned-docs", "vercel.json"))["framework"] is None
