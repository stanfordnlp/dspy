"""Tests for dspy.flex.dataset.DatasetStore (the persisted GEPA dataset)."""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.flex.dataset import DatasetStore


def _ex(q: str, a: str) -> dspy.Example:
    return dspy.Example(q=q, a=a).with_inputs("q")


def test_roundtrip_reuses_trainset_when_valset_is_same_object(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path, "Echo")
    trainset = [_ex("a", "b"), _ex("c", "d")]
    store.save(trainset, trainset)  # valset IS trainset

    loaded = store.load()
    assert loaded is not None
    lt, lv = loaded
    assert [e.q for e in lt] == ["a", "c"]
    assert [e.a for e in lt] == ["b", "d"]
    assert lt[0]._input_keys == {"q"}
    # A valset that was the same object as the trainset is stored as null and
    # reconstructed by reusing the trainset.
    assert lv is lt


def test_roundtrip_with_distinct_valset(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path, "Echo")
    trainset = [_ex("a", "b")]
    valset = [_ex("v", "w")]
    store.save(trainset, valset)

    lt, lv = store.load()
    assert [e.q for e in lt] == ["a"]
    assert lv is not lt
    assert [e.q for e in lv] == ["v"]


def test_input_keys_none_roundtrips_distinctly(tmp_path: Path) -> None:
    """An example with no inputs set must reload with _input_keys == None, not set()."""
    store = DatasetStore(tmp_path, "Echo")
    store.save([dspy.Example(q="x", a="y")], None)  # never called with_inputs

    lt, _ = store.load()
    assert lt[0]._input_keys is None


def test_load_returns_none_when_nothing_saved(tmp_path: Path) -> None:
    assert DatasetStore(tmp_path, "Echo").load() is None


def test_in_memory_mode_is_a_noop(tmp_path: Path) -> None:
    """root=None (in-memory Flex): save writes nothing and load returns None."""
    store = DatasetStore(None, "Echo")
    store.save([_ex("a", "b")], None)
    assert store.load() is None
    # No directory created anywhere relative to tmp_path either.
    assert not (tmp_path / ".flex").exists()


def test_save_writes_under_flex_dir(tmp_path: Path) -> None:
    DatasetStore(tmp_path, "Echo").save([_ex("a", "b")], None)
    assert (tmp_path / ".flex" / "Echo" / "dataset.json").exists()
