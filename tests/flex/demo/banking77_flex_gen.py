"""Implementation of a dspy.Flex module -- managed automatically.

This file starts life as a baseline that delegates to dspy.RLM, and is rewritten in place
by dspy.GEPA when you optimize the module (decomposing the task into predictors and Python code). 
It is a normal, runnable dspy.Module.

- You may edit the module class between the __FLEX_MODULE_BEGIN__/__FLEX_MODULE_END__ markers;
  on the next run dspy.Flex parses that class back out and runs your code as-is.
- __FLEX_SIGNATURE__ records the Signature this module was flexed from (for you and for GEPA).
- __FLEX_SIGNATURE_HASH__ guards against stale code: if you change the Signature, the hash no
  longer matches and dspy.Flex regenerates the baseline for the new Signature (re-run dspy.GEPA
  to re-optimize).

Leave the marker comments and the signature-hash line intact.
"""

# __FLEX_SIGNATURE_HASH__: c8d4322c3e040200c9b44ac4ee544a7ab3d2323cde445889240b4eddf7ffe867

# __FLEX_SIGNATURE_BEGIN__
# Signature: StringSignature
# Objective (docstring): You are an intent classifier for the BANKING77 dataset. Given a single retail-banking customer message, return the one most appropriate intent.
#
# The answer MUST be exactly one of these 77 snake_case intent labels:
# Refund_not_showing_up, activate_my_card, age_limit, apple_pay_or_google_pay, atm_support, automatic_top_up, balance_not_updated_after_bank_transfer, balance_not_updated_after_cheque_or_cash_deposit, beneficiary_not_allowed, cancel_transfer, card_about_to_expire, card_acceptance, card_arrival, card_delivery_estimate, card_linking, card_not_working, card_payment_fee_charged, card_payment_not_recognised, card_payment_wrong_exchange_rate, card_swallowed, cash_withdrawal_charge, cash_withdrawal_not_recognised, change_pin, compromised_card, contactless_not_working, country_support, declined_card_payment, declined_cash_withdrawal, declined_transfer, direct_debit_payment_not_recognised, disposable_card_limits, edit_personal_details, exchange_charge, exchange_rate, exchange_via_app, extra_charge_on_statement, failed_transfer, fiat_currency_support, get_disposable_virtual_card, get_physical_card, getting_spare_card, getting_virtual_card, lost_or_stolen_card, lost_or_stolen_phone, order_physical_card, passcode_forgotten, pending_card_payment, pending_cash_withdrawal, pending_top_up, pending_transfer, pin_blocked, receiving_money, request_refund, reverted_card_payment?, supported_cards_and_currencies, terminate_account, top_up_by_bank_transfer_charge, top_up_by_card_charge, top_up_by_cash_or_cheque, top_up_failed, top_up_limits, top_up_reverted, topping_up_by_card, transaction_charged_twice, transfer_fee_charged, transfer_into_account, transfer_not_received_by_recipient, transfer_timing, unable_to_verify_identity, verify_my_identity, verify_source_of_funds, verify_top_up, virtual_card_not_working, visa_or_mastercard, why_verify_identity, wrong_amount_of_cash_received, wrong_exchange_rate_for_cash_withdrawal.
#
# Return only the label, verbatim, with no extra words or punctuation.
# Input fields:
#   - text: str
# Output fields:
#   - intent: str
# __FLEX_SIGNATURE_END__

import dspy


# __FLEX_MODULE_BEGIN__
class StringSignatureModule(dspy.Module):
    def __init__(self):
        super().__init__()
        allowed_labels = [
            "Refund_not_showing_up", "activate_my_card", "age_limit", "apple_pay_or_google_pay",
            "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
            "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
            "card_delivery_estimate", "card_linking", "card_not_working",
            "card_payment_fee_charged", "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
            "contactless_not_working", "country_support", "declined_card_payment",
            "declined_cash_withdrawal", "declined_transfer",
            "direct_debit_payment_not_recognised", "disposable_card_limits",
            "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
            "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
            "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
            "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
            "order_physical_card", "passcode_forgotten", "pending_card_payment",
            "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
            "receiving_money", "request_refund", "reverted_card_payment?",
            "supported_cards_and_currencies", "terminate_account",
            "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
            "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
            "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
            "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
            "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
            "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
            "why_verify_identity", "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal",
        ]
        self._allowed_labels = allowed_labels
        labels_str = ", ".join(allowed_labels)

        instructions = (
            "You are an intent classifier for the BANKING77 dataset (retail-banking customer "
            "support messages). Given ONE customer message, return the single most appropriate "
            "intent label.\n\n"
            f"The answer MUST be exactly one of these 77 snake_case labels (verbatim, including "
            f"capitalization and any trailing '?'):\n{labels_str}\n\n"
            "How to think about it — focus on the customer's underlying GOAL / COMPLAINT, not "
            "surface keywords. For your top candidates, ask: what does the customer want resolved?\n\n"
            "Critical disambiguation rules (these are common error modes — read carefully):\n"
            "- A bank transfer was sent/received but the customer's BALANCE hasn't updated yet → "
            "  'balance_not_updated_after_bank_transfer'. This includes 'How long until a bank "
            "  transfer shows up on my balance?', 'Why hasn't my transfer appeared in my "
            "  balance?', 'When will a transfer reflect in my account?'. Do NOT use "
            "  'transfer_timing' for these. 'transfer_timing' is about how long the TRANSFER ITSELF "
            "  takes to complete/arrive at the recipient (outgoing transfer ETA), not about the "
            "  balance update.\n"
            "- Cheque/cash deposited but balance hasn't updated → "
            "  'balance_not_updated_after_cheque_or_cash_deposit'.\n"
            "- 'top_up_by_bank_transfer_charge' = fees for ADDING FUNDS to the account via bank "
            "  transfer. 'transfer_fee_charged' = fees for SENDING money out.\n"
            "- 'beneficiary_not_allowed' = a payee/beneficiary was rejected or can't be added. "
            "  'transfer_into_account' = receiving money into one's account / moving between own "
            "  accounts.\n"
            "- 'get_physical_card' = waiting for a physical card or PIN that hasn't arrived "
            "  ('I haven't received my PIN/card yet'). 'card_arrival' = asking expected delivery "
            "  time. 'order_physical_card' = requesting to order one. 'card_delivery_estimate' = "
            "  delivery time estimate question.\n"
            "- 'change_pin' = change/reset existing PIN. 'passcode_forgotten' = forgot app "
            "  passcode. 'pin_blocked' = PIN blocked after wrong attempts.\n"
            "- Unrecognized transaction: match the TYPE — 'card_payment_not_recognised' (card "
            "  purchase / vendor charge), 'cash_withdrawal_not_recognised' (ATM withdrawal), "
            "  'direct_debit_payment_not_recognised' (direct debit). Multiple unknown charges "
            "  suggesting theft → 'compromised_card'.\n"
            "- 'supported_cards_and_currencies' = which card brands/currencies supported. "
            "  'topping_up_by_card' = HOW to add funds with a card. 'top_up_by_card_charge' = "
            "  fees for topping up by card.\n"
            "- 'card_acceptance' = whether a merchant/place accepts the card. 'country_support' = "
            "  whether service works in a country.\n"
            "- 'Refund_not_showing_up' (note capital R) = refund expected but not appearing. "
            "  'request_refund' = asking to initiate a refund/dispute.\n"
            "- 'reverted_card_payment?' (note trailing ?) = a card payment was reversed.\n\n"
            "Return ONLY the label, verbatim from the list, with no extra words or punctuation."
        )

        self.classify = dspy.ChainOfThought(
            dspy.Signature("text -> intent", instructions)
        )

    def forward(self, **inputs):
        text = inputs.get("text", "")
        allowed = self._allowed_labels
        allowed_set = set(allowed)
        norm_map = {}
        for lbl in allowed:
            norm_map[lbl.lower().strip().rstrip("?")] = lbl

        raw = self.classify(text=text).intent
        candidate = (raw or "").strip().strip("`'\"").strip()

        if candidate in allowed_set:
            return dspy.Prediction(intent=candidate)

        if " " in candidate:
            best = None
            for lbl in allowed:
                if lbl in candidate and (best is None or len(lbl) > len(best)):
                    best = lbl
            if best is not None:
                return dspy.Prediction(intent=best)

        key = candidate.lower().strip().rstrip("?")
        if key in norm_map:
            return dspy.Prediction(intent=norm_map[key])

        cl = candidate.lower()
        for lbl in allowed:
            if lbl.lower() == cl or lbl.lower().rstrip("?") == cl.rstrip("?"):
                return dspy.Prediction(intent=lbl)

        return dspy.Prediction(intent=str(candidate) if candidate else allowed[0])
# __FLEX_MODULE_END__
