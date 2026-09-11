"""Guards in docs/scripts/sync_search_index.py against wiping a search store.

A file missing locally is deleted remotely, so an empty local set (a mistyped
``--site`` path, an empty build directory) must fail before any remote call.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs" / "scripts" / "sync_search_index.py"


@pytest.fixture(scope="module")
def sync():
    spec = importlib.util.spec_from_file_location("sync_search_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_client(remote):
    mxbai = MagicMock()
    mxbai.stores.files.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id=f"id:{rel}", external_id=rel, filename=rel, metadata={"sha256": "old"}) for rel in remote
        ]
    )
    return mxbai


def test_explicit_missing_site_dir_exits(sync, tmp_path):
    with pytest.raises(SystemExit):
        sync.docs_pages(str(tmp_path / "does-not-exist"))


def test_empty_site_dir_yields_no_pages(sync, tmp_path):
    assert sync.local_site_pages(tmp_path) == {}


@pytest.mark.parametrize("dry_run", [False, True])
def test_empty_local_set_refuses_before_any_remote_call(sync, dry_run):
    mxbai = fake_client(["a/index.md", "b/index.md"])
    with pytest.raises(SystemExit):
        sync.sync_store(mxbai, sync.DOCS_STORE, {}, dry_run=dry_run)
    assert not mxbai.mock_calls


def test_allow_empty_deletes_every_remote_file(sync):
    mxbai = fake_client(["a/index.md", "b/index.md"])
    sync.sync_store(mxbai, sync.DOCS_STORE, {}, dry_run=False, allow_empty=True)
    assert mxbai.stores.files.delete.call_count == 2


def test_stale_files_are_still_deleted_for_a_nonempty_set(sync):
    mxbai = fake_client(["a/index.md", "b/index.md"])
    sync.sync_store(mxbai, sync.DOCS_STORE, {"a/index.md": b"# a"}, dry_run=False)
    deleted = [call.args[0] for call in mxbai.stores.files.delete.call_args_list]
    assert deleted == ["id:b/index.md"]
