from __future__ import annotations

from pydantic import BaseModel

from dr_dspy.event_log import EventStore
from dr_dspy.run_metadata import REDACTED_VALUE, build_run_metadata


class ArgsModel(BaseModel):
    event_store: EventStore
    database_url: str | None


def test_build_run_metadata_redacts_and_serializes_pydantic_args() -> None:
    metadata = build_run_metadata(
        ArgsModel(
            event_store=EventStore.POSTGRES,
            database_url="postgres://user:pass@example/db",
        ),
        model_id="model/test",
        argv=["script.py", "--database-url", "postgres://user:pass@example/db"],
    )

    assert metadata["argv"] == ["script.py", "--database-url", REDACTED_VALUE]
    assert metadata["args"]["database_url"] == REDACTED_VALUE
    assert metadata["args"]["event_store"] == EventStore.POSTGRES.value
