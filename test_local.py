"""
Exercises the core logic directly (no HTTP, no network) so it can run in
environments without internet access. In a networked environment, run
`uvicorn bot:app` instead and use `judge_simulator.py` against it for the
full HTTP + LLM-judge test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from vera.composer import compose, extract_anchor_facts
from vera.context_store import ContextStore
from vera.conversation import (
    ConversationStore,
    detect_intent,
    detect_turn_language,
    is_confirmed_auto_reply,
    looks_like_auto_reply,
)
from vera.conversation_handlers import respond

DATASET = Path(__file__).parent / "dataset"


def load(kind: str, name: str) -> dict:
    return json.loads((DATASET / kind / f"{name}.json").read_text())


_KEY_FIELD = {
    "categories": "slug",
    "merchants": "merchant_id",
    "customers": "customer_id",
    "triggers": "id",
}


def load_all(kind: str) -> dict[str, dict]:
    out = {}
    key_field = _KEY_FIELD[kind]
    for f in (DATASET / kind).glob("*.json"):
        d = json.loads(f.read_text())
        out[d[key_field]] = d
    return out


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def test_context_store() -> None:
    section("1. ContextStore idempotency")
    store = ContextStore()
    ok, _ = store.put("merchant", "m_001", version=1, payload={"a": 1})
    assert ok, "first write should be accepted"
    ok, extra = store.put("merchant", "m_001", version=1, payload={"a": 2})
    assert not ok and extra["current_version"] == 1, "same version should be rejected"
    ok, _ = store.put("merchant", "m_001", version=2, payload={"a": 2})
    assert ok, "higher version should be accepted"
    assert store.get("merchant", "m_001")["a"] == 2
    print("PASS — put/idempotency/version-replace all behave per §2.1")


def test_conversation_signals() -> None:
    section("2. Auto-reply / intent / language detection")
    cases = [
        ("Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein hamari team tak pahuncha deti hoon.", True),
        ("Haan bilkul, batao kya karna hai", False),
        ("Thank you for contacting us, we will respond during business hours.", True),
    ]
    for text, expect_auto in cases:
        got = looks_like_auto_reply(text)
        status = "PASS" if got == expect_auto else "FAIL"
        print(f"{status} auto_reply={got} (expected {expect_auto}) :: {text[:60]}...")

    intent_cases = [
        ("Yes let's do it, go ahead", "accept"),
        ("Haan chalo kar do", "accept"),
        ("Not interested, please stop messaging", "decline"),
        ("What time do you close today?", "neutral"),
    ]
    for text, expected in intent_cases:
        got = detect_intent(text)
        status = "PASS" if got == expected else "FAIL"
        print(f"{status} intent={got} (expected {expected}) :: {text}")

    lang_cases = [
        ("Yes please, thank you", "en"),
        ("Haan bhai theek hai, kar do", "hi-en"),
        ("आपका बहुत धन्यवाद", "hi"),
    ]
    for text, expected in lang_cases:
        got = detect_turn_language(text)
        status = "PASS" if got == expected else "FAIL"
        print(f"{status} lang={got} (expected {expected}) :: {text}")


def test_compose_rule_based() -> None:
    section("3. compose() — rule-based fallback (no ANTHROPIC_API_KEY set here)")
    categories = load_all("categories")
    merchants = load_all("merchants")
    triggers = load_all("triggers")
    customers = load_all("customers")

    pairs = json.loads((DATASET / "test_pairs.json").read_text())["pairs"][:5]
    for pair in pairs:
        trigger = triggers[pair["trigger_id"]]
        merchant = merchants[pair["merchant_id"]]
        category = categories[merchant["category_slug"]]
        customer = customers.get(pair["customer_id"]) if pair["customer_id"] else None

        result = compose(category, merchant, trigger, customer)
        print(f"\n[{pair['test_id']}] trigger={trigger['kind']} merchant={merchant['identity']['name']}")
        print(f"  body: {result['body']}")
        print(f"  cta={result['cta']} send_as={result['send_as']}")
        print(f"  rationale: {result['rationale']}")


def test_multi_turn_flow() -> None:
    section("4. Multi-turn respond() — auto-reply exit + intent transition")
    categories = load_all("categories")
    merchants = load_all("merchants")
    triggers = load_all("triggers")

    trigger = list(triggers.values())[0]
    merchant = merchants[trigger["merchant_id"]]
    category = categories[merchant["category_slug"]]

    store = ConversationStore()
    state = store.get_or_create("conv_test_1", merchant["merchant_id"], None, trigger["id"])
    opening = compose(category, merchant, trigger, None, is_first_message=True)
    state.record_bot_message(opening["body"])
    print(f"[vera]     {opening['body']}")

    auto_reply_text = "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein hamari team tak pahuncha deti hoon."
    for i in range(3):
        result = respond(state, auto_reply_text, category=category, merchant=merchant, trigger=trigger)
        print(f"[merchant] {auto_reply_text}")
        print(f"[vera]     action={result['action']} :: {result.get('body') or result['rationale']}")
        if result["action"] == "end":
            print("PASS — bot exited gracefully after repeated auto-reply pattern")
            break

    section("4b. Intent transition (should skip re-qualification)")
    state2 = store.get_or_create("conv_test_2", merchant["merchant_id"], None, trigger["id"])
    opening2 = compose(category, merchant, trigger, None, is_first_message=True)
    state2.record_bot_message(opening2["body"])
    print(f"[vera]     {opening2['body']}")
    result2 = respond(state2, "Haan chalo, let's do it", category=category, merchant=merchant, trigger=trigger)
    print(f"[merchant] Haan chalo, let's do it")
    print(f"[vera]     action={result2['action']} :: {result2.get('body')}")
    print(f"  rationale: {result2.get('rationale')}")


if __name__ == "__main__":
    test_context_store()
    test_conversation_signals()
    test_compose_rule_based()
    test_multi_turn_flow()
    print("\nAll local checks completed.")
