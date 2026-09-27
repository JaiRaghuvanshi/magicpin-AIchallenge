"""
Vera-challenge bot server.

Implements the HTTP contract from challenge-testing-brief.md §2:
  POST /v1/context    - receive versioned context pushes (idempotent)
  POST /v1/tick       - periodic wake-up; bot may initiate proactive sends
  POST /v1/reply      - respond to a merchant/customer reply, synchronously
  GET  /v1/healthz    - liveness probe
  GET  /v1/metadata   - bot identity
  POST /v1/teardown   - optional; wipe state at end of test (per §11)

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-...      # optional — falls back to rule-based composer without it
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vera.composer import compose
from vera.context_store import ContextStore
from vera.conversation import ConversationStore, Stage
from vera.conversation_handlers import respond

app = FastAPI(title="Vera Challenge Bot")

START_TIME = time.time()
context_store = ContextStore()
conversation_store = ConversationStore()

MAX_ACTIONS_PER_TICK = 20  # per testing-brief §5

# suppression_key -> last-used simulated timestamp (string), to avoid re-firing
# the same campaign at the same merchant across ticks.
_used_suppression_keys: set[str] = set()
# (merchant_id, trigger_id) -> conversation_id, so we don't open two conversations
# for the same trigger.
_conv_by_merchant_trigger: dict[tuple[str, str], str] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ContextPush(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# /v1/context
# ---------------------------------------------------------------------------

@app.post("/v1/context")
async def push_context(body: ContextPush):
    accepted, extra = context_store.put(body.scope, body.context_id, body.version, body.payload)
    if not accepted:
        return {"accepted": False, "reason": "stale_version", **extra}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# /v1/tick
# ---------------------------------------------------------------------------

@app.post("/v1/tick")
async def tick(body: TickRequest):
    actions: list[dict] = []

    for trigger_id in body.available_triggers:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        trigger = context_store.get("trigger", trigger_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        merchant = context_store.get("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category = context_store.get("category", merchant.get("category_slug"))
        if not category:
            continue

        customer_id = trigger.get("customer_id")
        customer = context_store.get("customer", customer_id) if customer_id else None

        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and suppression_key in _used_suppression_keys:
            continue  # already messaged for this campaign/window — avoid spam

        conv_key = (merchant_id, trigger_id)
        if conv_key in _conv_by_merchant_trigger:
            continue  # already opened a conversation for this exact trigger

        conversation_id = f"conv_{merchant_id}_{trigger_id}"

        composed = compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            suppression_key=suppression_key,
            is_first_message=True,
        )

        state = conversation_store.get_or_create(conversation_id, merchant_id, customer_id, trigger_id)
        state.record_bot_message(composed["body"])

        if suppression_key:
            _used_suppression_keys.add(suppression_key)
        _conv_by_merchant_trigger[conv_key] = conversation_id

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# /v1/reply
# ---------------------------------------------------------------------------

@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    state = conversation_store.get(body.conversation_id)
    if state is None:
        # Judge replied to a conversation this bot instance doesn't remember
        # (e.g. restarted) — create a minimal state rather than erroring.
        state = conversation_store.get_or_create(
            body.conversation_id, body.merchant_id or "", body.customer_id
        )

    merchant = context_store.get("merchant", state.merchant_id) if state.merchant_id else None
    category = context_store.get("category", merchant.get("category_slug")) if merchant else None
    trigger = context_store.get("trigger", state.trigger_id) if state.trigger_id else None
    customer = context_store.get("customer", state.customer_id) if state.customer_id else None

    result = respond(
        state,
        body.message,
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
    )
    return result


# ---------------------------------------------------------------------------
# /v1/healthz, /v1/metadata, /v1/teardown
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": context_store.counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("VERA_TEAM_NAME", "Team Vera"),
        "team_members": os.environ.get("VERA_TEAM_MEMBERS", "").split(",") if os.environ.get("VERA_TEAM_MEMBERS") else [],
        "model": os.environ.get("VERA_LLM_MODEL", "claude-sonnet-4-6"),
        "approach": "4-context composer with trigger-family routing, rule-based fallback, "
                    "and a validation/repair pass for taboo-vocab, multi-CTA, and repeat detection.",
        "contact_email": os.environ.get("VERA_CONTACT_EMAIL", "team@example.com"),
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/teardown")
async def teardown():
    context_store.clear()
    conversation_store.clear()
    _used_suppression_keys.clear()
    _conv_by_merchant_trigger.clear()
    return {"status": "wiped"}
