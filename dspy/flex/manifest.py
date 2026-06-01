from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dspy.flex.exploration import FLEX_DIRNAME

MANIFEST_FILENAME = "manifest.json"


class ManifestStore:
    """Append-only ledger of accepted Flex module versions."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / FLEX_DIRNAME / MANIFEST_FILENAME

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"flex_modules": {}}
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("flex_modules", {})
        return data

    def latest(self, flex_id: str) -> dict[str, Any] | None:
        data = self.read()
        entry = data["flex_modules"].get(flex_id)
        if not entry or not entry.get("versions"):
            return None
        return entry["versions"][-1]

    def append_version(
        self,
        flex_id: str,
        src_path: str | Path,
        signature_hash: str,
        *,
        candidate_id: str | None = None,
        score: float | None = None,
        parents: list[str] | None = None,
        notes: str | None = None,
    ) -> int:
        """Append an accepted version.

        ``candidate_id`` and ``parents`` are the 12-char source hashes from
        :func:`dspy.flex.exploration.candidate_id` — they cross-link every
        manifest entry to its row in ``<flex_id>/exploration.jsonl``.
        """
        data = self.read()
        entry = data["flex_modules"].setdefault(flex_id, {"versions": []})
        next_id = (entry["versions"][-1]["id"] + 1) if entry["versions"] else 0
        version = {
            "id": next_id,
            "candidate_id": candidate_id,
            "src_path": str(src_path),
            "signature_hash": signature_hash,
            "score": score,
            "parents": parents or [],
            "ts": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        entry["versions"].append(version)
        self._write_atomic(data)
        return next_id

    def _write_atomic(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file in the same dir, then os.replace.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json.tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
