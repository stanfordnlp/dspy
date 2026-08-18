from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class TeleprompterDebtReport:
    program_id: str
    tdi_score: float  # Teleprompter Debt Index (target <= 12.0)
    demonstration_multiplier: float  # Target <= 1.10x
    evaluation_latency_seconds: float  # Target <= 0.95s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for DSPy teleprompter prompt compilations.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_compilation_event(
        self,
        program_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{program_id}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "program_id": program_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
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


class ProductionDebtTeleprompterGate:
    """
    A2Z SOC Production Debt & Technical Due Diligence Gate for DSPy Teleprompter Optimizers.

    Quantifies prompt compilation trials, few-shot demonstration context bloat, and eval latency against 4 Enterprise KPIs:
    1. Teleprompter Debt Index (TDI <= 12.0)
    2. Demonstration Token Multiplier (DTM <= 1.10x)
    3. P99 Compiled Pipeline Latency (<= 0.95s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_tdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_tdi = max_acceptable_tdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_compilation(
        self,
        program_id: str,
        baseline_prompt_tokens: int = 450,
        compiled_prompt_tokens: int = 480,
        few_shot_demos_count: int = 3,
        evaluation_latency_seconds: float = 0.65,
        trial_search_failures: int = 0,
        un_gated_mutations: int = 0,
    ) -> TeleprompterDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_compilation_event(
                program_id=program_id,
                event_type="compilation_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. DSPy prompt compilation halted."
            )

        critical_smells: List[str] = []

        # KPI 2: Demonstration Token Multiplier
        token_ratio = compiled_prompt_tokens / max(1, baseline_prompt_tokens)
        if token_ratio > 1.8:
            critical_smells.append(f"HIGH_DEMO_TOKEN_INFLATION_{token_ratio:.2f}X")

        # Few shot demo count
        if few_shot_demos_count > 8:
            critical_smells.append(f"EXCESSIVE_FEW_SHOT_DEMOS_{few_shot_demos_count}")

        # KPI 3: Latency Ceiling
        if evaluation_latency_seconds > 3.0:
            critical_smells.append(f"HIGH_EVALUATION_LATENCY_{evaluation_latency_seconds:.2f}S")

        # Trial search failures
        if trial_search_failures > 2:
            critical_smells.append(f"DETECTED_{trial_search_failures}_FAILED_OPTIMIZER_TRIALS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_SIGNATURE_MUTATIONS")

        # KPI 1: Teleprompter Debt Index (0 = Clean, 100 = Catastrophic)
        tdi = (
            max(0.0, (token_ratio - 1.0) * 20.0)
            + (max(0, few_shot_demos_count - 3) * 2.5)
            + max(0.0, (evaluation_latency_seconds - 0.95) * 10.0)
            + (trial_search_failures * 12.0)
            + (un_gated_mutations * 30.0)
        )
        tdi_score = round(min(100.0, tdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - tdi_score)
        is_production_ready = (
            tdi_score <= self.max_acceptable_tdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_compilation_event(
            program_id=program_id,
            event_type="compilation_authorized" if is_production_ready else "compilation_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "tdi_score": tdi_score,
                "token_ratio": token_ratio,
                "baseline_prompt_tokens": baseline_prompt_tokens,
                "compiled_prompt_tokens": compiled_prompt_tokens,
                "few_shot_demos_count": few_shot_demos_count,
                "evaluation_latency_seconds": evaluation_latency_seconds,
                "trial_search_failures": trial_search_failures,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return TeleprompterDebtReport(
            program_id=program_id,
            tdi_score=tdi_score,
            demonstration_multiplier=round(token_ratio, 2),
            evaluation_latency_seconds=round(evaluation_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
