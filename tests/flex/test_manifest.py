"""Tests for the ManifestStore append-only ledger."""

from __future__ import annotations

import json

from dspy.flex.manifest import ManifestStore


def test_initial_read_returns_empty_structure(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    data = store.read()
    assert data == {"flex_modules": {}}


def test_append_creates_first_version(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    vid = store.append_version("myflex", tmp_path / "src.py", "abc123")
    assert vid == 0
    data = store.read()
    assert "myflex" in data["flex_modules"]
    versions = data["flex_modules"]["myflex"]["versions"]
    assert len(versions) == 1
    assert versions[0]["id"] == 0
    assert versions[0]["signature_hash"] == "abc123"


def test_append_increments_version_id(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    store.append_version("myflex", tmp_path / "src.py", "abc", candidate_id="seedhash00aa")
    second_id = store.append_version(
        "myflex",
        tmp_path / "src.py",
        "abc",
        candidate_id="childhash00b",
        score=0.7,
        parents=["seedhash00aa"],
    )
    assert second_id == 1
    latest = store.latest("myflex")
    assert latest is not None
    assert latest["id"] == 1
    assert latest["candidate_id"] == "childhash00b"
    assert latest["score"] == 0.7
    assert latest["parents"] == ["seedhash00aa"]


def test_latest_returns_none_when_missing(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    assert store.latest("missing") is None


def test_manifest_is_valid_json(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    store.append_version("a", tmp_path / "x.py", "h1")
    store.append_version("b", tmp_path / "y.py", "h2")
    raw = (tmp_path / ".flex" / "manifest.json").read_text()
    parsed = json.loads(raw)
    assert set(parsed["flex_modules"].keys()) == {"a", "b"}
