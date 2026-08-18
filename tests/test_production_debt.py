import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../dspy/teleprompt/production_debt.py",
)
spec = importlib.util.spec_from_file_location("dspy_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["dspy_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtTeleprompterGate = production_debt_mod.ProductionDebtTeleprompterGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtTeleprompterGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtTeleprompterGate(
            never_equate_intent_to_approval=True,
            max_acceptable_tdi=12.0,
        )

    def test_clean_compilation_passes_readiness(self) -> None:
        report = self.gate.evaluate_compilation(
            program_id="mipro_v2_enterprise_rag_pipeline",
            baseline_prompt_tokens=450,
            compiled_prompt_tokens=480,
            few_shot_demos_count=3,
            evaluation_latency_seconds=0.65,
            trial_search_failures=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.tdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_compilation_fails_debt(self) -> None:
        report = self.gate.evaluate_compilation(
            program_id="unconstrained_trial_search",
            baseline_prompt_tokens=400,
            compiled_prompt_tokens=1400,  # High token inflation (3.5x)
            few_shot_demos_count=12,  # Excessive few-shot demos
            evaluation_latency_seconds=5.5,  # High evaluation latency
            trial_search_failures=4,  # 4 failed optimizer trials
            un_gated_mutations=2,  # 2 un-gated signature mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.tdi_score, 50.0)
        self.assertIn("HIGH_DEMO_TOKEN_INFLATION_3.50X", report.critical_smells)
        self.assertIn("EXCESSIVE_FEW_SHOT_DEMOS_12", report.critical_smells)
        self.assertIn("HIGH_EVALUATION_LATENCY_5.50S", report.critical_smells)
        self.assertIn("DETECTED_4_FAILED_OPTIMIZER_TRIALS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_SIGNATURE_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_compilation("program-1")
        self.gate.evaluate_compilation("program-2")
        self.gate.evaluate_compilation("program-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
