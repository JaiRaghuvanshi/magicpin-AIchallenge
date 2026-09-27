"""
Optional deliverable per challenge-brief.md §7.4:

    def respond(state: ConversationState, merchant_message: str) -> dict

Implements the /v1/reply behavior from challenge-testing-brief.md §2.3 and
the open challenges from challenge-brief.md §12 (auto-reply detection,
intent-transition handling, graceful exit).

Signature note: the brief's `respond(state, merchant_message)` doesn't carry
enough to *compose* a follow-up (no category/merchant/trigger data), so this
implementation accepts those as keyword args with the first two positional
params preserved exactly as specified. bot.py always calls it with the full
bundle; a caller that only has `state` + the message text can still call it
for pure state-machine decisions (auto-reply / intent), just without a
freshly composed follow-up body.
"""

from __future__ import annotations

from typing import Literal

from .composer import compose
from .conversation import (
    ConversationState,
    Stage,
    detect_intent,
    detect_turn_language,
    is_confirmed_auto_reply,
    looks_like_auto_reply,
)
from .llm_providers import LLMProvider

ReplyAction = Literal["send", "wait", "end"]


def respond(
    state: ConversationState,
    merchant_message: str,
    *,
    category: dict | None = None,
    merchant: dict | None = None,
    trigger: dict | None = None,
    customer: dict | None = None,
    provider: LLMProvider | None = None,
) -> dict:
    from_role = "customer" if state.customer_id else "merchant"
    state.record_incoming(from_role, merchant_message)  # type: ignore[arg-type]

    # --- 1. Auto-reply detection (open challenge #1) -----------------------
    if looks_like_auto_reply(merchant_message) or is_confirmed_auto_reply(state):
        if state.auto_reply_streak >= 2 or _already_probed_auto_reply(state):
            state.stage = Stage.ENDED
            return {
                "action": "end",
                "rationale": "Confirmed auto-reply (repeated/canned pattern); "
                              "exiting gracefully rather than burning further turns.",
            }
        # First time we see it: one gentle probe (Pattern B in the brief), then stop if it repeats.
        probe_body = _auto_reply_probe(merchant, detect_turn_language(merchant_message))
        state.record_bot_message(probe_body)
        return {
            "action": "send",
            "body": probe_body,
            "cta": "open_ended",
            "rationale": "Message reads as a canned auto-reply; probing once for a human "
                         "before treating it as confirmed and exiting.",
        }

    # --- 2. Explicit decline -> graceful exit (open challenge #5) ----------
    intent = detect_intent(merchant_message)
    if intent == "decline":
        state.stage = Stage.ENDED
        return {
            "action": "end",
            "rationale": "Merchant signaled not interested; exiting without further nudges.",
        }

    # --- 3. Explicit accept -> skip qualification, go to action mode -------
    if intent == "accept":
        state.stage = Stage.ACTION

    # --- 4. Nudge budget exhausted (open challenge #5) ----------------------
    if state.should_stop():
        state.stage = Stage.ENDED
        return {
            "action": "end",
            "rationale": "3 unanswered nudges without a substantive reply; stopping to avoid spam.",
        }

    # --- 5. Compose the next turn ------------------------------------------
    if category and merchant and trigger:
        composed = compose(
            category=category,
            merchant=merchant,
            trigger=_augment_trigger_for_followup(trigger, state, intent),
            customer=customer,
            suppression_key=trigger.get("suppression_key"),
            is_first_message=False,
            previously_sent={t.body.strip().lower() for t in state.turns if t.from_role == "vera"},
            provider=provider,
        )
        state.record_bot_message(composed["body"])
        return {
            "action": "send",
            "body": composed["body"],
            "cta": composed["cta"],
            "rationale": composed["rationale"],
        }

    # No context bundle available — can't compose; wait rather than send something generic.
    return {
        "action": "wait",
        "wait_seconds": 300,
        "rationale": "No category/merchant/trigger context available yet to compose a reply.",
    }


def _already_probed_auto_reply(state: ConversationState) -> bool:
    return any("2 minute" in t.body.lower() or "2-min" in t.body.lower() or "khud dekhna" in t.body.lower()
               for t in state.turns if t.from_role == "vera")


def _auto_reply_probe(merchant: dict | None, lang: str) -> str:
    name = ""
    if merchant:
        name = merchant.get("identity", {}).get("owner_first_name", "")
    if lang in ("hi", "hi-en"):
        return (
            f"Samajh gayi{', ' + name if name else ''}. Team tak pahunchane se pehle, "
            f"kya aap khud 2 minute mein dekhna chahenge ki exactly kya karna hai?"
        )
    return (
        f"Got it{', ' + name if name else ''}. Before this goes to your team — want to "
        f"take 2 minutes yourself to see exactly what's needed?"
    )


def _augment_trigger_for_followup(trigger: dict, state: ConversationState, intent: str) -> dict:
    """Give the composer a hint that this is a follow-up, and whether the
    merchant just accepted (so it should move straight to action, per
    Pattern D's anti-pattern of re-qualifying after a yes)."""
    augmented = dict(trigger)
    payload = dict(augmented.get("payload", {}))
    payload["_conversation_stage"] = state.stage.value
    payload["_merchant_just_accepted"] = intent == "accept"
    payload["_turn_number"] = len(state.turns)
    augmented["payload"] = payload
    return augmented
