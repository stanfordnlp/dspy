from __future__ import annotations

import json
import textwrap

import dspy
from dspy.flex import flex
from dspy.flex.exploration import ExplorationStore, candidate_id
from dspy.utils.dummies import DummyLM

PREDICTORS_SRC = textwrap.dedent("""
    PREDICTORS = {"echo": dspy.Predict("q -> a")}
""").strip()

FORWARD_SRC = textwrap.dedent("""
    def forward(self, q):
        return dspy.Prediction(a=self.echo(q=q).a)
""").strip()


# --- Unit tests for the store primitives -------------------------------------


def test_candidate_id_is_deterministic_and_short() -> None:
    a = candidate_id("p1", "f1")
    b = candidate_id("p1", "f1")
    c = candidate_id("p1", "f2")
    assert a == b
    assert a != c
    assert len(a) == 12


def test_exploration_store_records_event_and_writes_candidate(tmp_path) -> None:
    store = ExplorationStore(tmp_path, "myflex")
    cid = store.record(
        "codegen",
        predictors_src="PREDICTORS = {}",
        forward_src="def forward(self):\n    return 1",
        signature_hash="abc",
    )
    assert cid is not None
    assert store.has_candidate(cid)
    cand = store.get_candidate(cid)
    assert cand is not None
    assert cand["predictors_src"] == "PREDICTORS = {}"
    assert cand["signature_hash"] == "abc"


def test_exploration_store_deduplicates_candidate_files(tmp_path) -> None:
    store = ExplorationStore(tmp_path, "myflex")
    store.record("codegen", predictors_src="P", forward_src="F", signature_hash="h")
    store.record("evaluate", predictors_src="P", forward_src="F", score=0.5)
    store.record("evaluate", predictors_src="P", forward_src="F", score=0.7)
    # One candidate file, three events.
    assert len(store.list_candidates()) == 1
    history = store.get_history()
    assert [e["event"] for e in history] == ["codegen", "evaluate", "evaluate"]
    assert history[2]["score"] == 0.7


def test_exploration_store_best_score(tmp_path) -> None:
    store = ExplorationStore(tmp_path, "myflex")
    store.record("evaluate", predictors_src="P1", forward_src="F1", score=0.2)
    store.record("evaluate", predictors_src="P2", forward_src="F2", score=0.9)
    store.record("evaluate", predictors_src="P3", forward_src="F3", score=0.5)
    best = store.best_score()
    assert best is not None
    assert best[1] == 0.9
    expected_cid = candidate_id("P2", "F2")
    assert best[0] == expected_cid


def test_exploration_store_handles_events_without_sources(tmp_path) -> None:
    store = ExplorationStore(tmp_path, "myflex")
    cid = store.record("load")
    assert cid is None
    assert store.get_history()[0]["event"] == "load"


def test_exploration_log_is_jsonl(tmp_path) -> None:
    store = ExplorationStore(tmp_path, "myflex")
    store.record("codegen", predictors_src="P", forward_src="F", signature_hash="h")
    store.record("evaluate", predictors_src="P", forward_src="F", score=0.3)
    text = (tmp_path / ".flex" / "myflex" / "exploration.jsonl").read_text()
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line parses


def test_event_rows_do_not_carry_signature_hash(tmp_path) -> None:
    """signature_hash is captured on candidate files, not on each event row."""
    store = ExplorationStore(tmp_path, "myflex")
    store.record("codegen", predictors_src="P", forward_src="F", signature_hash="sighash")
    store.record("evaluate", predictors_src="P", forward_src="F", score=0.5)
    for entry in store.get_history():
        assert "signature_hash" not in entry
    # ...but it IS captured once on the candidate file.
    cid = candidate_id("P", "F")
    cand = store.get_candidate(cid)
    assert cand is not None
    assert cand["signature_hash"] == "sighash"


def test_exploration_store_with_none_root_is_a_noop() -> None:
    """In-memory Flex mode passes root=None; all writes must silently no-op."""
    store = ExplorationStore(None, "myflex")
    cid = store.record(
        "codegen",
        predictors_src="P",
        forward_src="F",
    )
    assert cid is None
    assert store.get_history() == []
    assert store.list_candidates() == []
    assert store.has_candidate("anything") is False
    assert store.best_score() is None


# --- Integration tests: Flex codegen writes into .flex/ next to persist_to ---


def test_codegen_records_event_in_flex_directory(tmp_path) -> None:
    dspy.configure(lm=DummyLM([{"predictors_src": PREDICTORS_SRC, "forward_src": FORWARD_SRC}]))

    @flex(persist_to=str(tmp_path / "echo_flex.py"))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()

    flex_dir = tmp_path / ".flex" / "Echo"
    assert flex_dir.exists()
    history = ExplorationStore(tmp_path, "Echo").get_history()
    events = [e["event"] for e in history]
    assert "codegen" in events
    assert "accept" in events  # because persist_to was set


def test_reload_from_disk_records_load_event(tmp_path) -> None:
    dspy.configure(lm=DummyLM([{"predictors_src": PREDICTORS_SRC, "forward_src": FORWARD_SRC}]))

    @flex(persist_to=str(tmp_path / "echo_flex.py"))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()
    # Second construction should hit the on-disk cache and record a "load" event.
    Echo()

    events = [e["event"] for e in ExplorationStore(tmp_path, "Echo").get_history()]
    assert events.count("codegen") == 1
    assert events.count("load") == 1


def test_in_memory_flex_writes_no_exploration_files(tmp_path, monkeypatch) -> None:
    """When persist_to is None, no .flex/ directory is created anywhere."""
    monkeypatch.chdir(tmp_path)
    dspy.configure(lm=DummyLM([{"predictors_src": PREDICTORS_SRC, "forward_src": FORWARD_SRC}]))

    @flex
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()
    assert not (tmp_path / ".flex").exists()
