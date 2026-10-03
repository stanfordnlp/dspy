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
        "<p>Documentation example for DSPy 3.4.0b1</p>"
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
    publish_site(
        repository=repository,
        site=make_site(tmp_path / "beta", "3.0.0b1"),
        identifier="3.0.0b1",
        aliases=[],
        package_source="workflow-wheel",
    )

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
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", "versioned-docs:3.0.0b1"], cwd=repository, capture_output=True
        ).returncode
        != 0
    )


@requires_mike
def test_mike_refuses_to_replace_an_immutable_snapshot(tmp_path):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "first", "original")
    for path in ("diving-deeper/tools", "diving-deeper/tools-react-and-mcp"):
        page = site / path
        page.mkdir(parents=True)
        (page / "index.html").write_text(path)
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


@requires_mike
@pytest.mark.parametrize("branch", ["versioned-docs", "master"])
def test_stable_publication_prunes_only_matching_prereleases_atomically(tmp_path, branch):
    repository = make_repository(tmp_path)
    superseded = ["3.4.0a1", "3.4.0b1", "3.4.0b12", "3.4.0rc2"]
    preserved = ["current", "3.3.1", "3.3.0rc1", "3.4.1b1", "3.5.0b1", "3.4.00b1"]
    for version in preserved + superseded:
        publish_site(
            repository=repository,
            site=make_site(tmp_path / version, version),
            identifier=version,
            aliases=["3.3"] if version == "3.3.1" else [],
            package_source="pypi-wheel",
            branch=branch,
        )
    before = subprocess.check_output(["git", "rev-parse", branch], cwd=repository)
    trees = {
        version: subprocess.check_output(["git", "rev-parse", f"{branch}:{version}"], cwd=repository)
        for version in preserved + ["3.3"]
    }
    inventory_before = json.loads(branch_file(repository, branch, "versions.json"))
    arguments = {
        "repository": repository,
        "site": make_site(tmp_path / "stable", "stable"),
        "identifier": "3.4.0",
        "aliases": ["3.4", "latest"],
        "package_source": "workflow-wheel",
        "branch": branch,
        "required_current_renderer": "zensical",
    }

    assert publish_site(**arguments)

    assert subprocess.check_output(["git", "rev-parse", f"{branch}^"], cwd=repository) == before
    assert "stable" in branch_file(repository, branch, "3.4.0/index.html")
    assert "../3.4.0/" in branch_file(repository, branch, "3.4/index.html")
    assert "../../3.4.0/guide/" in branch_file(repository, branch, "3.4/guide/index.html")
    assert "../3.4.0/" in branch_file(repository, branch, "latest/index.html")
    assert "../../3.4.0/guide/" in branch_file(repository, branch, "latest/guide/index.html")
    inventory = json.loads(branch_file(repository, branch, "versions.json"))
    assert set(next(entry for entry in inventory if entry["version"] == "3.4.0")["aliases"]) == {"3.4", "latest"}
    assert [entry for entry in inventory if entry["version"] in preserved] == [
        entry for entry in inventory_before if entry["version"] in preserved
    ]
    for version in superseded:
        assert version not in {entry["version"] for entry in inventory}
        assert (
            subprocess.run(
                ["git", "cat-file", "-e", f"{branch}:{version}"], cwd=repository, capture_output=True
            ).returncode
            != 0
        )
    for version, tree in trees.items():
        assert subprocess.check_output(["git", "rev-parse", f"{branch}:{version}"], cwd=repository) == tree
    after = subprocess.check_output(["git", "rev-parse", branch], cwd=repository)
    assert not publish_site(**arguments)
    assert subprocess.check_output(["git", "rev-parse", branch], cwd=repository) == after


@requires_mike
@pytest.mark.parametrize("newer_patch", [False, True])
def test_stable_cleanup_preserves_existing_snapshot_and_alias_on_retry(tmp_path, newer_patch):
    repository = make_repository(tmp_path)
    site = make_site(tmp_path / "stable", "original")
    arguments = {
        "repository": repository,
        "site": site,
        "identifier": "3.4.0",
        "aliases": ["3.4"],
        "package_source": "workflow-wheel",
    }
    assert publish_site(**arguments)
    if newer_patch:
        publish_site(**{**arguments, "site": make_site(tmp_path / "newer", "newer"), "identifier": "3.4.1"})
    # Model a deployment produced before pruning was introduced.
    publish_site(**{**arguments, "site": make_site(tmp_path / "beta", "beta"), "identifier": "3.4.0b1", "aliases": []})
    before = subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository)
    stable_tree = subprocess.check_output(["git", "rev-parse", "versioned-docs:3.4.0"], cwd=repository)
    alias_tree = subprocess.check_output(["git", "rev-parse", "versioned-docs:3.4"], cwd=repository)
    inventory = json.loads(branch_file(repository, "versioned-docs", "versions.json"))

    original = (site / "index.html").read_bytes()
    (site / "index.html").write_text("different")
    with pytest.raises(RuntimeError, match="immutable Mike snapshot"):
        publish_site(**arguments)
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository) == before
    (site / "index.html").write_bytes(original)

    assert publish_site(**arguments)
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs^"], cwd=repository) == before
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs:3.4.0"], cwd=repository) == stable_tree
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs:3.4"], cwd=repository) == alias_tree
    assert json.loads(branch_file(repository, "versioned-docs", "versions.json")) == [
        entry for entry in inventory if entry["version"] != "3.4.0b1"
    ]
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", "versioned-docs:3.4.0b1"], cwd=repository, capture_output=True
        ).returncode
        != 0
    )
    after = subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository)
    assert not publish_site(**arguments)
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository) == after


@requires_mike
@pytest.mark.parametrize(
    "fault",
    ["missing-inventory", "malformed-inventory", "not-a-list", "aliased-candidate", "malformed-aliases"],
)
def test_stable_pruning_requires_inventory_and_unaliased_candidates(tmp_path, fault):
    from mike import git_utils

    from docs.scripts.publish_versioned_docs import working_directory

    repository = make_repository(tmp_path)
    arguments = {
        "repository": repository,
        "site": make_site(tmp_path / "beta", "beta"),
        "identifier": "3.4.0b1",
        "aliases": [],
        "package_source": "workflow-wheel",
    }
    publish_site(**arguments)
    inventory = json.loads(branch_file(repository, "versioned-docs", "versions.json"))
    with working_directory(repository), git_utils.Commit("versioned-docs", "Damage pruning inventory") as commit:
        if fault == "aliased-candidate":
            inventory[0]["aliases"] = ["preview"]
        elif fault == "malformed-aliases":
            inventory[0]["aliases"] = ""
        if fault == "missing-inventory":
            commit.delete_files(["versions.json"])
        else:
            text = {"malformed-inventory": "{", "not-a-list": "{}"}.get(fault, json.dumps(inventory))
            commit.add_file(git_utils.FileInfo("versions.json", text))
    before = subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository)

    with pytest.raises(RuntimeError, match=r"inventory|unexpectedly own aliases"):
        publish_site(**{**arguments, "identifier": "3.4.0", "aliases": ["3.4"]})
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository) == before
    if fault == "malformed-aliases":
        assert "beta" in branch_file(repository, "versioned-docs", "3.4.0b1/index.html")
        assert json.loads(branch_file(repository, "versioned-docs", "versions.json")) == inventory


@requires_mike
def test_stable_pruning_aborts_the_entire_commit_on_write_failure(tmp_path, monkeypatch):
    from mike import git_utils

    repository = make_repository(tmp_path)
    arguments = {
        "repository": repository,
        "site": make_site(tmp_path / "beta", "beta"),
        "identifier": "3.4.0b1",
        "aliases": [],
        "package_source": "workflow-wheel",
    }
    publish_site(**arguments)
    before = subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository)
    with pytest.raises(RuntimeError, match=r"missing index\.html"):
        publish_site(**{**arguments, "identifier": "3.4.0", "site": tmp_path / "missing-site"})
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository) == before
    add_file = git_utils.Commit.add_file

    def fail_inventory_write(commit, file):
        if file.path == "versions.json":
            raise OSError("simulated inventory write failure")
        add_file(commit, file)

    stable = {**arguments, "identifier": "3.4.0", "aliases": ["3.4"]}
    with monkeypatch.context() as patch:
        patch.setattr(git_utils.Commit, "add_file", fail_inventory_write)
        with pytest.raises(OSError, match="simulated inventory write failure"):
            publish_site(**stable)
    assert subprocess.check_output(["git", "rev-parse", "versioned-docs"], cwd=repository) == before
    assert publish_site(**stable)
    assert not publish_site(**stable)
