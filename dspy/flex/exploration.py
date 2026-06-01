from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

FLEX_DIRNAME = ".flex"

EventType = Literal["codegen", "load", "evaluate", "propose", "accept"]
"""Closed set of event names that may appear in ``exploration.jsonl``.

- ``codegen``: a Flex module's implementation was generated from scratch by the codegen LM.
- ``load``: a Flex module loaded its implementation from a persisted ``.py`` file.
- ``evaluate``: a candidate was scored on a batch of examples. Score is the batch mean.
- ``propose``: the reflection LM produced a *non-trivial* revision of one component
- ``accept``: a candidate was written to disk and appended as a new manifest version.
"""


def candidate_id(predictors_src: str, forward_src: str) -> str:
    """Deterministic short ID for a ``(predictors_src, forward_src)`` pair."""
    blob = f"{predictors_src}\n----\n{forward_src}".encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class ExplorationStore:
    """Append-only candidate + event store for a single Flex module.

    Lives at ``<root>/.flex/<flex_id>/``. When ``root`` is ``None`` (in-memory
    Flex mode), every read returns the empty value and every write is a no-op
    — callers don't have to null-check.
    """

    def __init__(self, root: str | Path | None, flex_id: str):
        self.flex_id = flex_id
        self.root: Path | None = (Path(root) / FLEX_DIRNAME / flex_id) if root is not None else None
        self.candidates_dir: Path | None = (self.root / "candidates") if self.root is not None else None
        self.log_path: Path | None = (self.root / "exploration.jsonl") if self.root is not None else None

    def has_candidate(self, cid: str) -> bool:
        if self.candidates_dir is None:
            return False
        return (self.candidates_dir / f"{cid}.json").exists()

    def get_candidate(self, cid: str) -> dict[str, Any] | None:
        if self.candidates_dir is None:
            return None
        p = self.candidates_dir / f"{cid}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list_candidates(self) -> list[str]:
        if self.candidates_dir is None or not self.candidates_dir.exists():
            return []
        return sorted(p.stem for p in self.candidates_dir.glob("*.json"))

    def get_history(self) -> list[dict[str, Any]]:
        if self.log_path is None or not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def best_score(self) -> tuple[str, float] | None:
        """Return the (candidate_id, score) with the highest seen score."""
        best: tuple[str, float] | None = None
        for entry in self.get_history():
            score = entry.get("score")
            cid = entry.get("candidate_id")
            if score is None or cid is None:
                continue
            if best is None or score > best[1]:
                best = (cid, float(score))
        return best

    def record(
        self,
        event: EventType,
        *,
        predictors_src: str | None = None,
        forward_src: str | None = None,
        signature_hash: str | None = None,
        score: float | None = None,
        parents: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Append one event row to ``exploration.jsonl``.

        ``signature_hash`` is captured only on the *candidate file* (first-seen
        snapshot); it is not embedded in each event row since it doesn't change
        within a compile run. Reconstruct it from ``candidates/<cid>.json`` when
        needed.
        """
        if self.root is None:
            return None
        assert self.candidates_dir is not None and self.log_path is not None

        cid: str | None = None
        if predictors_src is not None and forward_src is not None:
            cid = candidate_id(predictors_src, forward_src)
            if not self.has_candidate(cid):
                self.candidates_dir.mkdir(parents=True, exist_ok=True)
                cand_payload = {
                    "candidate_id": cid,
                    "signature_hash": signature_hash,
                    "predictors_src": predictors_src,
                    "forward_src": forward_src,
                    "first_seen_ts": _now_iso(),
                    "first_seen_event": event,
                }
                _atomic_write_json(self.candidates_dir / f"{cid}.json", cand_payload)

        entry: dict[str, Any] = {"event": event, "ts": _now_iso()}
        if cid is not None:
            entry["candidate_id"] = cid
        if score is not None:
            entry["score"] = score
        if parents is not None:
            entry["parents"] = parents
        if extra:
            for k, v in extra.items():
                if k not in entry:
                    entry[k] = v

        self._append_log(entry)
        return cid

    def _append_log(self, entry: dict[str, Any]) -> None:
        if self.root is None or self.log_path is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".cand-", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
