from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FLEX_DIRNAME = ".flex"
_PROJECT_ROOT_MARKERS = (".git", "pyproject.toml", "setup.py", "setup.cfg")


def find_project_root(start: str | Path | None = None) -> Path:
    cur = Path(start).resolve() if start else Path.cwd().resolve()
    while True:
        for marker in _PROJECT_ROOT_MARKERS:
            if (cur / marker).exists():
                return cur
        if cur == cur.parent:
            break
        cur = cur.parent
    return Path(start).resolve() if start else Path.cwd().resolve()


def candidate_id(predictors_src: str, forward_src: str) -> str:
    """Deterministic short ID for a ``(predictors_src, forward_src)`` pair."""
    blob = f"{predictors_src}\n----\n{forward_src}".encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class ExplorationStore:
    """Append-only candidate + event store for a single Flex module."""

    def __init__(self, root: str | Path, flex_id: str):
        self.root = Path(root) / FLEX_DIRNAME / flex_id
        self.candidates_dir = self.root / "candidates"
        self.log_path = self.root / "exploration.jsonl"

    def has_candidate(self, cid: str) -> bool:
        return (self.candidates_dir / f"{cid}.json").exists()

    def get_candidate(self, cid: str) -> dict[str, Any] | None:
        p = self.candidates_dir / f"{cid}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list_candidates(self) -> list[str]:
        if not self.candidates_dir.exists():
            return []
        return sorted(p.stem for p in self.candidates_dir.glob("*.json"))

    def get_history(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
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
        event: str,
        *,
        predictors_src: str | None = None,
        forward_src: str | None = None,
        signature_hash: str | None = None,
        score: float | None = None,
        parents: list[str] | None = None,
        feedback_summary: str | None = None,
        rejected_reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
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
        if signature_hash is not None:
            entry["signature_hash"] = signature_hash
        if score is not None:
            entry["score"] = score
        if parents is not None:
            entry["parents"] = parents
        if feedback_summary is not None:
            entry["feedback_summary"] = feedback_summary
        if rejected_reason is not None:
            entry["rejected_reason"] = rejected_reason
        if extra:
            for k, v in extra.items():
                if k not in entry:
                    entry[k] = v

        self._append_log(entry)
        return cid

    def _append_log(self, entry: dict[str, Any]) -> None:
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
