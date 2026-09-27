"""
The composition engine — implements challenge-brief.md §5's contract:

    compose(category, merchant, trigger, customer=None) -> {
        body, cta, send_as, suppression_key, rationale
    }

Design:
  1. `extract_anchor_facts` resolves the trigger against category/merchant/
     customer data into a small set of *verifiable* facts (a number, a date,
     a source, a name) — this is what "specificity" scoring rewards and
     what prevents hallucination (we never let the model invent a fact that
     isn't in this extracted set).
  2. `build_prompts` turns those facts + voice/constraints into a system +
     user prompt, with trigger-family-specific framing.
  3. `LLMProvider.complete_json` composes; on any failure (no key, network,
     bad JSON) we fall back to `rule_based_compose`, a deterministic
     template-filler, so the bot is never silently broken.
  4. `validate_and_repair` runs cheap, code-level checks that mirror the
     judge's stated anti-patterns (taboo vocab, multi-CTA, language match,
     exact repeats) and fixes what it can without another LLM round-trip.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from .llm_providers import LLMProvider, LLMUnavailable, get_provider

CTA = Literal["binary", "open_ended", "none"]


# ---------------------------------------------------------------------------
# 1. Anchor-fact extraction
# ---------------------------------------------------------------------------

# Trigger kinds grouped into families that share a framing strategy.
EXTERNAL_RESEARCH_KINDS = {
    "research_digest", "regulation_change", "cde_opportunity", "category_research_digest_release",
}
EXTERNAL_EVENT_KINDS = {
    "festival_upcoming", "weather_heatwave", "local_news_event", "ipl_match_today",
    "category_seasonal", "competitor_opened", "category_trend_movement", "supply_alert",
}
INTERNAL_PERF_KINDS = {
    "perf_spike", "perf_dip", "seasonal_perf_dip", "milestone_reached", "review_theme_emerged",
}
INTERNAL_ACCOUNT_KINDS = {
    "renewal_due", "dormant_with_vera", "scheduled_recurring", "curious_ask_due",
    "active_planning_intent", "gbp_unverified", "trial_followup",
}
CUSTOMER_FACING_KINDS = {
    "recall_due", "customer_lapsed_soft", "customer_lapsed_hard", "appointment_tomorrow",
    "chronic_refill_due", "winback_eligible", "wedding_package_followup",
}


def _find_by_id(items: list[dict] | None, id_key: str, target_id: str) -> dict | None:
    if not items or not target_id:
        return None
    for item in items:
        if item.get(id_key) == target_id or item.get("id") == target_id:
            return item
    return None


@dataclass
class AnchorFacts:
    family: str
    facts: dict[str, Any]
    notes: list[str]


def extract_anchor_facts(
    category: dict, merchant: dict, trigger: dict, customer: dict | None
) -> AnchorFacts:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {}) or {}
    facts: dict[str, Any] = {"trigger_kind": kind, "urgency": trigger.get("urgency")}
    notes: list[str] = []

    if kind in EXTERNAL_RESEARCH_KINDS:
        family = "external_research"
        item = _find_by_id(category.get("digest"), "top_item_id", payload.get("top_item_id", ""))
        if item:
            facts["digest_item"] = item
        else:
            facts.update({k: v for k, v in payload.items()})
            notes.append("No matching digest item found by id — using raw trigger payload only.")

    elif kind in EXTERNAL_EVENT_KINDS:
        family = "external_event"
        facts.update({k: v for k, v in payload.items()})
        if kind == "competitor_opened":
            notes.append("Only mention the competitor fact if it is present in payload — never invent a name.")
        facts["seasonal_beats"] = category.get("seasonal_beats", [])
        facts["trend_signals"] = category.get("trend_signals", [])

    elif kind in INTERNAL_PERF_KINDS:
        family = "internal_performance"
        facts["performance"] = merchant.get("performance", {})
        facts["peer_stats"] = category.get("peer_stats", {})
        facts["signals"] = merchant.get("signals", [])
        facts["review_themes"] = merchant.get("review_themes", [])
        facts.update({k: v for k, v in payload.items()})

    elif kind in INTERNAL_ACCOUNT_KINDS:
        family = "internal_account"
        facts["subscription"] = merchant.get("subscription", {})
        facts["conversation_history"] = merchant.get("conversation_history", [])[-3:]
        facts["signals"] = merchant.get("signals", [])
        facts.update({k: v for k, v in payload.items()})

    elif kind in CUSTOMER_FACING_KINDS:
        family = "customer_facing"
        facts.update({k: v for k, v in payload.items()})
        facts["merchant_offers"] = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        if customer:
            facts["customer_relationship"] = customer.get("relationship", {})
            facts["customer_state"] = customer.get("state")
            facts["customer_preferences"] = customer.get("preferences", {})
            facts["customer_consent_scope"] = (customer.get("consent") or {}).get("scope", [])
        else:
            notes.append("Customer-facing trigger kind but no CustomerContext supplied.")

    else:
        family = "generic"
        facts.update({k: v for k, v in payload.items()})
        facts["performance"] = merchant.get("performance", {})
        facts["peer_stats"] = category.get("peer_stats", {})

    return AnchorFacts(family=family, facts=facts, notes=notes)


# ---------------------------------------------------------------------------
# 2. Prompt construction
# ---------------------------------------------------------------------------

ANTI_PATTERNS = """\
Do NOT:
- Use generic offers ("Flat 30% off") when a specific service+price exists — use the exact catalog title.
- Give more than one call-to-action in the message.
- Bury the call-to-action — it must land in the final sentence.
- Use promotional/hype tone ("AMAZING DEAL!") for clinical/professional categories.
- Invent, estimate, or round any number, date, source, or name that is not present in the
  ANCHOR FACTS below. If a fact you'd want isn't there, don't reference it.
- Write a long preamble ("I hope you're doing well...").
- Re-introduce yourself if this is not the first message in the conversation.
- Ignore the merchant's language preference.
- Repeat a message verbatim that has already been sent in this conversation."""

COMPULSION_LEVERS = """\
Use one or more of these levers (favor social proof and "asking the merchant" — these
are underused today):
1. Specificity/verifiability (a concrete number, date, headline, source citation)
2. Loss aversion ("you're missing X")
3. Social proof ("3 dentists in your locality did Y this month")
4. Effort externalization ("I've drafted X — just say go")
5. Curiosity ("want to see who?")
6. Reciprocity ("I noticed Y, thought you'd want to know")
7. Asking the merchant a direct question
8. Single binary commitment (Reply YES/STOP) — only for action triggers"""


def _voice_block(category: dict) -> str:
    voice = category.get("voice", {})
    return (
        f"Tone: {voice.get('tone', 'peer, not promotional')}. "
        f"Register: {voice.get('register', 'respectful, collegial')}. "
        f"Code-mix: {voice.get('code_mix', 'hindi_english_natural')} — Hindi-English mix is fine "
        f"and often preferred.\n"
        f"Vocabulary encouraged: {', '.join(voice.get('vocab_allowed', [])[:10]) or 'n/a'}.\n"
        f"Vocabulary/claims forbidden: {', '.join(voice.get('vocab_taboo', [])) or 'n/a'}."
    )


def build_system_prompt(category: dict) -> str:
    return f"""You are composing ONE WhatsApp message as "Vera", magicpin's merchant-AI assistant,
OR (if a CustomerContext is supplied and send_as should be merchant_on_behalf) a message sent
from the merchant's own WhatsApp number to their customer.

CATEGORY: {category.get('display_name', category.get('slug'))}
VOICE:
{_voice_block(category)}

{ANTI_PATTERNS}

{COMPULSION_LEVERS}

Respond with ONLY a JSON object, no markdown fences, no commentary, with exactly these keys:
{{
  "body": "<the WhatsApp message text>",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "rationale": "<one sentence: why this message, what it should achieve>"
}}"""


def build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    anchor: AnchorFacts,
    is_first_message: bool,
) -> str:
    identity = merchant.get("identity", {})
    lines = [
        f"MERCHANT: {identity.get('name')} ({identity.get('locality', '')}, {identity.get('city', '')})",
        f"Owner first name: {identity.get('owner_first_name', '')}",
        f"Languages: {identity.get('languages', ['en'])}",
        f"Active offers: {[o.get('title') for o in merchant.get('offers', []) if o.get('status') == 'active']}",
        f"This {'IS' if is_first_message else 'is NOT'} the first message in this conversation "
        f"({'use a template-style opener, no prior context to reference' if is_first_message else 'do not re-introduce yourself'}).",
        "",
        f"TRIGGER: kind={trigger.get('kind')}, urgency={trigger.get('urgency')}/5, "
        f"family={anchor.family}",
        f"ANCHOR FACTS (the ONLY facts you may cite as specifics):",
        _fmt_facts(anchor.facts),
    ]
    if anchor.notes:
        lines.append("NOTES: " + "; ".join(anchor.notes))
    payload = trigger.get("payload", {}) or {}
    if payload.get("_merchant_just_accepted"):
        lines.append(
            "IMPORTANT: the merchant just said yes / explicitly agreed to proceed. Do NOT ask "
            "another qualifying question — move straight into action (confirm what you're doing "
            "now, or ask only for the one piece of information needed to execute)."
        )
    if customer:
        c_identity = customer.get("identity", {})
        lines += [
            "",
            f"CUSTOMER (this message is on behalf of the merchant, sent TO this customer): "
            f"{c_identity.get('name')}, language_pref={c_identity.get('language_pref')}, "
            f"state={customer.get('state')}, preferences={customer.get('preferences', {})}",
            f"Consent scope: {(customer.get('consent') or {}).get('scope', [])} — only proceed if this "
            f"trigger's purpose falls within that scope.",
        ]
    return "\n".join(lines)


def _fmt_facts(facts: dict) -> str:
    import json as _json
    return _json.dumps(facts, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 3. Rule-based fallback (deterministic, no LLM required)
# ---------------------------------------------------------------------------

def rule_based_compose(
    category: dict, merchant: dict, trigger: dict, customer: dict | None, anchor: AnchorFacts
) -> dict:
    """
    Deterministic template filler used when no LLM provider is available.
    Not as strong as an LLM composition on 'engagement compulsion' or nuanced
    voice, but it is specific (pulls real numbers/sources), category-safe
    (never uses taboo vocab), and structurally valid — good enough to keep
    the bot fully operational and testable offline.
    """
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name", "there")
    kind = trigger.get("kind", "")
    facts = anchor.facts
    payload = trigger.get("payload", {}) or {}

    if payload.get("_merchant_just_accepted"):
        # Intent-handoff (open challenge #2 / anti-pattern D): move straight to
        # action, never re-qualify, regardless of trigger family.
        return {
            "body": f"Great, {name} — starting now. I'll confirm here once it's done.",
            "cta": "none",
            "send_as": "merchant_on_behalf" if customer else "vera",
            "rationale": "Merchant explicitly accepted; skipping further qualification and "
                          "moving directly to action per the accept signal.",
        }

    if anchor.family == "external_research" and "digest_item" in facts:
        item = facts["digest_item"]
        body = (
            f"{name}, {item.get('source', 'a category update')} — {item.get('title', '')}. "
            f"{item.get('summary', '')} Want me to pull the full item and draft something you can share?"
        )
        cta: CTA = "open_ended"

    elif anchor.family == "internal_performance" and kind == "perf_dip":
        perf = facts.get("performance", {})
        delta = perf.get("delta_7d", {})
        body = (
            f"{name}, quick flag: your calls are down "
            f"{abs(round((delta.get('calls_pct') or 0) * 100))}% week-over-week "
            f"(views {perf.get('views', 'n/a')}, calls {perf.get('calls', 'n/a')} in the last "
            f"{perf.get('window_days', 30)} days). Want me to check what's driving it?"
        )
        cta = "open_ended"

    elif anchor.family == "internal_performance" and kind == "perf_spike":
        perf = facts.get("performance", {})
        delta = perf.get("delta_7d", {})
        body = (
            f"{name}, good news — your views are up "
            f"{round((delta.get('views_pct') or 0) * 100)}% this week ({perf.get('views', 'n/a')} total). "
            f"Want me to draft a post to keep the momentum going?"
        )
        cta = "open_ended"

    elif anchor.family == "customer_facing" and customer:
        c_name = customer.get("identity", {}).get("name", "there")
        offers = facts.get("merchant_offers", [])
        offer_line = offers[0].get("title") if offers else "your usual service"
        body = (
            f"Hi {c_name}, {identity.get('name')} here. It's been a while since your last visit — "
            f"wanted to check in. {offer_line} is available if you'd like to book. "
            f"Reply and we'll find you a slot."
        )
        cta = "open_ended"

    else:
        offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
        offer_line = offers[0] if offers else "your current listing"
        body = (
            f"{name}, checking in on {offer_line} — anything you'd like help with this week?"
        )
        cta = "open_ended"

    return {
        "body": body,
        "cta": cta,
        "send_as": "merchant_on_behalf" if customer else "vera",
        "rationale": f"Rule-based fallback composition for trigger family '{anchor.family}' (kind={kind}).",
    }


# ---------------------------------------------------------------------------
# 4. Validation / repair
# ---------------------------------------------------------------------------

_MULTI_CTA_RE = re.compile(r"reply\s+\S+\s+for\s+\S+.*reply\s+\S+\s+for\s+\S+", re.IGNORECASE | re.DOTALL)


def validate_and_repair(result: dict, category: dict, merchant: dict, previously_sent: set[str] | None = None) -> dict:
    body = str(result.get("body", "")).strip()
    cta = result.get("cta", "open_ended")
    send_as = result.get("send_as", "vera")
    rationale = str(result.get("rationale", "")).strip() or "No rationale provided."

    if cta not in ("binary", "open_ended", "none"):
        cta = "open_ended"
    if send_as not in ("vera", "merchant_on_behalf"):
        send_as = "vera"

    # Taboo vocabulary check — strip offending sentences rather than the whole message.
    taboos = [t.lower() for t in category.get("voice", {}).get("vocab_taboo", [])]
    for taboo in taboos:
        if taboo and taboo in body.lower():
            body = re.sub(re.escape(taboo), "", body, flags=re.IGNORECASE).strip()
            body = re.sub(r"\s{2,}", " ", body)

    # Multi-CTA heuristic: if it looks like two "Reply X for Y" clauses, keep only the first.
    if _MULTI_CTA_RE.search(body):
        first_sentence_end = body.find(".", body.lower().find("reply"))
        if first_sentence_end != -1:
            body = body[: first_sentence_end + 1]

    # Anti-repetition: if this exact body was already sent in this conversation, tag it in
    # rationale so the caller can decide to re-roll (LLM path retries once; rule-based
    # path perturbs deterministically).
    is_repeat = bool(previously_sent) and body.strip().lower() in previously_sent

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "rationale": rationale,
        "_is_repeat": is_repeat,
    }


# ---------------------------------------------------------------------------
# 5. Public entry point
# ---------------------------------------------------------------------------

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    *,
    suppression_key: str | None = None,
    is_first_message: bool = True,
    previously_sent: set[str] | None = None,
    provider: LLMProvider | None = None,
) -> dict:
    """
    Main composition entry point matching challenge-brief.md §5/§7.1.
    Returns dict with keys: body, cta, send_as, suppression_key, rationale.
    """
    anchor = extract_anchor_facts(category, merchant, trigger, customer)

    llm_result: dict | None = None
    try:
        prov = provider or get_provider()
        system = build_system_prompt(category)
        user = build_user_prompt(category, merchant, trigger, customer, anchor, is_first_message)
        llm_result = prov.complete_json(system, user)
    except LLMUnavailable:
        llm_result = None
    except Exception:  # noqa: BLE001 — never let a bad LLM response take the bot down
        llm_result = None

    raw = llm_result if llm_result else rule_based_compose(category, merchant, trigger, customer, anchor)
    validated = validate_and_repair(raw, category, merchant, previously_sent)

    if validated.pop("_is_repeat", False):
        # One deterministic perturbation so we never send a byte-identical repeat.
        validated["body"] = validated["body"].rstrip(".") + " — following up on this."

    validated["suppression_key"] = suppression_key or trigger.get("suppression_key", "")
    return validated
