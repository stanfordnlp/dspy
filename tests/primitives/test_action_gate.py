import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../dspy/primitives/action_gate.py",
)
spec = importlib.util.spec_from_file_location("dspy_action_gate", file_path)
action_gate_mod = importlib.util.module_from_spec(spec)
sys.modules["dspy_action_gate"] = action_gate_mod
spec.loader.exec_module(action_gate_mod)

ActionGateAssertion = action_gate_mod.ActionGateAssertion
ActionBoundaryAssertionException = action_gate_mod.ActionBoundaryAssertionException
DSPyActionLedger = action_gate_mod.DSPyActionLedger
GENESIS_HASH = action_gate_mod.GENESIS_HASH


class TestDSPyActionGate(unittest.TestCase):
    def setUp(self):
        self.gate = ActionGateAssertion(
            never_equate_intent_to_approval=True,
            enforce_action_boundary=True,
        )

    def test_assert_action_boundary_passed(self):
        # Should execute cleanly without error
        self.gate.assert_action_boundary(
            condition=True,
            tool_name="rag_search",
            metadata={"query": "A2Z SOC"},
        )
        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "passed")

    def test_assert_action_boundary_failed_raises_exception(self):
        with self.assertRaises(ActionBoundaryAssertionException):
            self.gate.assert_action_boundary(
                condition=False,
                tool_name="database_drop",
                message="Unauthorized database drop attempted in compiled LM program.",
            )

    def test_suggest_action_boundary(self):
        res = self.gate.suggest_action_boundary(
            condition=False,
            tool_name="costly_web_scrape",
            message="Warning: High token consumption path detected.",
        )
        self.assertFalse(res)
        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "warning")

    def test_hash_chain_integrity(self):
        ledger = DSPyActionLedger()
        ledger.record_program_step("s1", "t1", "ok", {"step": 1})
        ledger.record_program_step("s2", "t2", "ok", {"step": 2})
        ledger.record_program_step("s3", "t3", "ok", {"step": 3})

        entries = ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
