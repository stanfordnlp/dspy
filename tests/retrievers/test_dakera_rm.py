"""Tests for DakeraRM — Dakera memory retrieval module.

Tests use ``unittest.mock`` to intercept HTTP calls, so no live server is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dspy.retrievers.dakera_rm import DakeraRM

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SEARCH_RESPONSE = {
    "memories": [
        {
            "memory": {
                "id": "mem_001",
                "content": "The project deadline is next Friday.",
                "agent_id": "test-agent",
            },
            "score": 0.92,
        },
        {
            "memory": {
                "id": "mem_002",
                "content": "Sprint review is on Thursday.",
                "agent_id": "test-agent",
            },
            "score": 0.78,
        },
    ]
}

STORE_RESPONSE = {
    "id": "mem_003",
    "content": "New memory content.",
    "agent_id": "test-agent",
    "created_at": "2026-07-01T00:00:00Z",
}

FORGET_RESPONSE = {"deleted": ["mem_001"]}


def make_rm(**kwargs) -> DakeraRM:
    """Create a DakeraRM with test defaults."""
    defaults = {
        "agent_id": "test-agent",
        "url": "http://localhost:3000",
        "api_key": "test-key",
        "k": 5,
    }
    defaults.update(kwargs)
    return DakeraRM(**defaults)


def mock_post(response_json: dict):
    """Return a mock for requests.post that yields response_json."""
    mock_response = MagicMock()
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()
    return MagicMock(return_value=mock_response)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


def test_constructor_explicit_args():
    rm = make_rm(session_id="sess-1", timeout=30.0)
    assert rm.agent_id == "test-agent"
    assert rm.url == "http://localhost:3000"
    assert rm.k == 5
    assert rm.session_id == "sess-1"
    assert rm.timeout == 30.0


def test_constructor_url_trailing_slash_stripped():
    rm = DakeraRM(agent_id="a", url="http://localhost:3000/", api_key="key")
    assert rm.url == "http://localhost:3000"


def test_constructor_env_fallback(monkeypatch):
    monkeypatch.setenv("DAKERA_URL", "http://dakera.example.com")
    monkeypatch.setenv("DAKERA_API_KEY", "env-key")
    rm = DakeraRM(agent_id="env-agent")
    assert rm.url == "http://dakera.example.com"


def test_constructor_default_url_when_no_env(monkeypatch):
    monkeypatch.delenv("DAKERA_URL", raising=False)
    monkeypatch.setenv("DAKERA_API_KEY", "env-key")
    rm = DakeraRM(agent_id="env-agent")
    assert rm.url == "http://localhost:3000"


def test_constructor_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        DakeraRM(agent_id="test-agent")


# ---------------------------------------------------------------------------
# forward() / __call__ tests
# ---------------------------------------------------------------------------


def test_forward_returns_prediction_with_passages():
    rm = make_rm()
    with patch("requests.post", mock_post(SEARCH_RESPONSE)):
        result = rm("What are the deadlines?")

    assert hasattr(result, "passages")
    assert hasattr(result, "memory_ids")
    assert hasattr(result, "scores")
    assert result.passages == [
        "The project deadline is next Friday.",
        "Sprint review is on Thursday.",
    ]
    assert result.memory_ids == ["mem_001", "mem_002"]
    assert result.scores == pytest.approx([0.92, 0.78])


def test_forward_respects_k_override():
    rm = make_rm(k=3)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(SEARCH_RESPONSE)()

    with patch("requests.post", fake_post):
        rm("query", k=1)

    assert captured["payload"]["top_k"] == 1


def test_forward_uses_instance_k_when_not_overridden():
    rm = make_rm(k=7)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(SEARCH_RESPONSE)()

    with patch("requests.post", fake_post):
        rm("query")

    assert captured["payload"]["top_k"] == 7


def test_forward_sends_session_id_when_set():
    rm = make_rm(session_id="sess-abc")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(SEARCH_RESPONSE)()

    with patch("requests.post", fake_post):
        rm("query")

    assert captured["payload"]["session_id"] == "sess-abc"


def test_forward_omits_session_id_when_none():
    rm = make_rm()
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(SEARCH_RESPONSE)()

    with patch("requests.post", fake_post):
        rm("query")

    assert "session_id" not in captured["payload"]


def test_forward_per_call_session_id_overrides_instance():
    rm = make_rm(session_id="instance-sess")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(SEARCH_RESPONSE)()

    with patch("requests.post", fake_post):
        rm("query", session_id="call-sess")

    assert captured["payload"]["session_id"] == "call-sess"


def test_forward_multi_query_deduplication():
    """Results with duplicate memory IDs across queries must appear only once."""
    rm = make_rm(k=2)
    # Both queries return the same two memories.
    with patch("requests.post", mock_post(SEARCH_RESPONSE)):
        result = rm(["deadline?", "schedule?"])

    assert len(result.passages) == 2  # deduplicated
    assert result.memory_ids == ["mem_001", "mem_002"]


def test_forward_multi_query_list():
    rm = make_rm(k=5)

    # Return different memories per query
    first_response = {
        "memories": [
            {"memory": {"id": "mem_A", "content": "Alpha."}, "score": 0.9},
        ]
    }
    second_response = {
        "memories": [
            {"memory": {"id": "mem_B", "content": "Beta."}, "score": 0.8},
        ]
    }
    call_count = 0

    def alternating_post(url, json=None, headers=None, timeout=None):
        nonlocal call_count
        resp = first_response if call_count == 0 else second_response
        call_count += 1
        return mock_post(resp)()

    with patch("requests.post", alternating_post):
        result = rm(["query A", "query B"])

    assert result.passages == ["Alpha.", "Beta."]
    assert result.memory_ids == ["mem_A", "mem_B"]


def test_forward_sends_bearer_token():
    rm = make_rm()
    captured_headers = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured_headers.update(headers or {})
        return mock_post(SEARCH_RESPONSE)()

    with patch("requests.post", fake_post):
        rm("query")

    assert captured_headers.get("Authorization") == "Bearer test-key"


def test_forward_empty_response():
    rm = make_rm()
    with patch("requests.post", mock_post({"memories": []})):
        result = rm("obscure query")

    assert result.passages == []
    assert result.memory_ids == []
    assert result.scores == []


def test_forward_http_error_propagates():
    import requests as req

    rm = make_rm()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req.HTTPError("403 Forbidden")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(req.HTTPError):
            rm("query")


# ---------------------------------------------------------------------------
# store() tests
# ---------------------------------------------------------------------------


def test_store_sends_correct_payload():
    rm = make_rm()
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return mock_post(STORE_RESPONSE)()

    with patch("requests.post", fake_post):
        resp = rm.store(
            "New memory content.",
            tags=["tag1"],
            metadata={"source": "test"},
        )

    assert captured["url"].endswith("/v1/memory/store")
    assert captured["payload"]["content"] == "New memory content."
    assert captured["payload"]["agent_id"] == "test-agent"
    assert captured["payload"]["tags"] == ["tag1"]
    assert captured["payload"]["metadata"] == {"source": "test"}
    assert resp == STORE_RESPONSE


def test_store_session_id_override():
    rm = make_rm(session_id="default-sess")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(STORE_RESPONSE)()

    with patch("requests.post", fake_post):
        rm.store("text", session_id="override-sess")

    assert captured["payload"]["session_id"] == "override-sess"


def test_store_omits_optional_fields_when_none():
    rm = make_rm()
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return mock_post(STORE_RESPONSE)()

    with patch("requests.post", fake_post):
        rm.store("text")

    assert "session_id" not in captured["payload"]
    assert "tags" not in captured["payload"]
    assert "metadata" not in captured["payload"]


# ---------------------------------------------------------------------------
# forget() tests
# ---------------------------------------------------------------------------


def test_forget_sends_correct_payload():
    rm = make_rm()
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return mock_post(FORGET_RESPONSE)()

    with patch("requests.post", fake_post):
        resp = rm.forget(["mem_001", "mem_002"])

    assert captured["url"].endswith("/v1/memory/forget")
    assert captured["payload"]["agent_id"] == "test-agent"
    assert captured["payload"]["memory_ids"] == ["mem_001", "mem_002"]
    assert resp == FORGET_RESPONSE


# ---------------------------------------------------------------------------
# dump_state / load_state tests
# ---------------------------------------------------------------------------


def test_dump_state_includes_expected_keys():
    rm = make_rm(k=7, session_id="sess-x", timeout=20.0)
    state = rm.dump_state()
    assert state["k"] == 7
    assert state["agent_id"] == "test-agent"
    assert state["url"] == "http://localhost:3000"
    assert state["session_id"] == "sess-x"
    assert state["timeout"] == 20.0
    # api_key must NOT appear in persisted state
    assert "api_key" not in state
    assert "_api_key" not in state


def test_load_state_restores_fields():
    rm = make_rm()
    state = {
        "k": 10,
        "agent_id": "loaded-agent",
        "url": "http://remote:3000",
        "session_id": "s42",
        "timeout": 5.0,
    }
    rm.load_state(state)
    assert rm.k == 10
    assert rm.agent_id == "loaded-agent"
    assert rm.url == "http://remote:3000"
    assert rm.session_id == "s42"
    assert rm.timeout == 5.0


def test_load_state_defaults_timeout_when_missing():
    rm = make_rm()
    rm.load_state(
        {
            "k": 3,
            "agent_id": "a",
            "url": "http://localhost:3000",
        }
    )
    assert rm.timeout == 10.0
