from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


class ActionBoundaryAssertionException(Exception):
    """Raised when an action boundary assertion is violated during LM program execution."""
    pass


class ActionGateAssertion:
    """
    A2Z SOC ActionGate Assertion & Zero-Trust Boundary for DSPy compiled LM programs.
    Enforces never_equate_intent_to_approval, emergency kill-switches, and cryptographic receipts.
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        enforce_action_boundary: bool = True,
    ):
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.enforce_action_boundary = enforce_action_boundary
        self.ledger = DSPyActionLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for p in (Path("artifacts/KILL"), Path("/tmp/KILL")):
            if p.exists():
                return True
        return False

    def assert_action_boundary(
        self,
        condition: bool,
        tool_name: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Hard constraint assertion: Raises ActionBoundaryAssertionException if condition fails or kill-switch is engaged.
        """
        if self.check_kill_switch():
            self.ledger.record_program_step(
                step_name="kill_switch_halt",
                tool_name=tool_name,
                status="halted",
                metadata={"reason": "emergency_kill_switch_active"},
            )
            raise ActionBoundaryAssertionException(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. Execution halted."
            )

        if not condition:
            err_msg = message or f"ActionBoundary assertion failed for tool '{tool_name}' (never_equate_intent_to_approval violated)."
            self.ledger.record_program_step(
                step_name="assertion_failure",
                tool_name=tool_name,
                status="failed",
                metadata={"error": err_msg, **(metadata or {})},
            )
            raise ActionBoundaryAssertionException(err_msg)

        self.ledger.record_program_step(
            step_name="assertion_passed",
            tool_name=tool_name,
            status="passed",
            metadata=metadata or {},
        )

    def suggest_action_boundary(
        self,
        condition: bool,
        tool_name: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Soft constraint suggestion: Logs warning to cryptographic ledger without halting execution.
        """
        if not condition:
            warn_msg = message or f"ActionBoundary suggestion warning for tool '{tool_name}'."
            log.warning(warn_msg)
            self.ledger.record_program_step(
                step_name="suggestion_warning",
                tool_name=tool_name,
                status="warning",
                metadata={"warning": warn_msg, **(metadata or {})},
            )
            return False

        self.ledger.record_program_step(
            step_name="suggestion_passed",
            tool_name=tool_name,
            status="passed",
            metadata=metadata or {},
        )
        return True


class DSPyActionLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for DSPy LM programs and prompt compilation runs.
    """

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self._last_hash = GENESIS_HASH

    def record_program_step(
        self,
        step_name: str,
        tool_name: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{step_name}|{tool_name}|{status}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "step_name": step_name,
            "tool_name": tool_name,
            "status": status,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True
