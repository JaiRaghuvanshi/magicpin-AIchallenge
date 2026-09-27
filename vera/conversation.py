"""
Per-conversation state + the "open challenges" from challenge-brief.md §12:

1. Auto-reply detection      -> same verbatim merchant message 3+ times = auto-reply
2. Intent-transition handling -> explicit "yes/let's do it" skips qualification, goes to action mode
3. Multi-turn cadence         -> track unanswered nudges, back off / stop appropriately
4. Language detection         -> lightweight per-turn hi/en/mix classification
5. Knowing when to stop       -> graceful exit on "not interested" or 3 unanswered nudges
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal


class Stage(str, Enum):
    PITCH = "pitch"          # still qualifying / pitching
    ACTION = "action"        # merchant said yes -> doing the thing now
    ENDED = "ended"          # conversation closed (not-interested / exhausted nudges / auto-reply exit)


@dataclass
class Turn:
    from_role: Literal["merchant", "customer", "vera"]
    body: str
    ts: str


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    trigger_id: str | None = None
    stage: Stage = Stage.PITCH
    turns: list[Turn] = field(default_factory=list)
    unanswered_bot_nudges: int = 0     # consecutive bot messages sent with no reply in between
    auto_reply_streak: int = 0         # consecutive identical merchant replies
    last_merchant_body_norm: str | None = None
    sent_bodies: set[str] = field(default_factory=set)  # anti-repetition (exact dedupe)

    def record_bot_message(self, body: str) -> None:
        self.turns.append(Turn(from_role="vera", body=body, ts=_now()))
        self.sent_bodies.add(_norm(body))
        self.unanswered_bot_nudges += 1

    def record_incoming(self, from_role: Literal["merchant", "customer"], body: str) -> None:
        self.turns.append(Turn(from_role=from_role, body=body, ts=_now()))
        self.unanswered_bot_nudges = 0

        norm = _norm(body)
        if from_role == "merchant":
            if self.last_merchant_body_norm is not None and norm == self.last_merchant_body_norm:
                self.auto_reply_streak += 1
            else:
                self.auto_reply_streak = 1
            self.last_merchant_body_norm = norm

    def is_repeat(self, candidate_body: str) -> bool:
        return _norm(candidate_body) in self.sent_bodies

    def should_stop(self) -> bool:
        return self.stage == Stage.ENDED or self.unanswered_bot_nudges >= 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Auto-reply detection
# ---------------------------------------------------------------------------

_AUTO_REPLY_MARKERS = [
    "automated assistant",
    "auto-reply",
    "thank you for contacting",
    "aapki jaankari ke liye",
    "hamari team tak pahuncha",
    "team tak pahuncha",
    "we will get back to you",
    "we'll get back to you",
    "currently unavailable",
    "business hours",
    "will respond shortly",
]


def looks_like_auto_reply(body: str) -> bool:
    norm = _norm(body)
    return any(marker in norm for marker in _AUTO_REPLY_MARKERS)


def is_confirmed_auto_reply(state: ConversationState) -> bool:
    """Per brief hint: same message verbatim 3+ times = auto-reply. We also
    fast-path on a single strong lexical marker, since production Vera's
    stated weakness is burning 2-3 turns before it commits to that call."""
    if state.auto_reply_streak >= 3:
        return True
    if state.last_merchant_body_norm and looks_like_auto_reply(state.last_merchant_body_norm):
        return True
    return False


# ---------------------------------------------------------------------------
# Intent detection (Pattern D anti-pattern: don't re-qualify after a "yes")
# ---------------------------------------------------------------------------

_INTENT_YES_PATTERNS = [
    r"\byes\b", r"\bok(ay)?\b", r"\bsure\b", r"\blet'?s do it\b", r"\bgo ahead\b",
    r"\bsign me up\b", r"\bstart\b", r"\bchalo\b", r"\bhaan\b", r"\bkar do\b",
    r"\bjoin(na)? (karna|karni|karta) hai\b", r"\bjudna hai\b", r"\bkijiye\b",
    r"\bproceed\b", r"\bhaan bhai\b",
]

_INTENT_NO_PATTERNS = [
    r"\bnot interested\b", r"\bno thanks\b", r"\bstop\b", r"\bnahi chahiye\b",
    r"\bnahin chahiye\b", r"\bmat karo\b", r"\bunsubscribe\b", r"\bnever mind\b",
    r"\bleave (it|us) alone\b",
]


def detect_intent(body: str) -> Literal["accept", "decline", "neutral"]:
    norm = _norm(body)
    if any(re.search(p, norm) for p in _INTENT_NO_PATTERNS):
        return "decline"
    if any(re.search(p, norm) for p in _INTENT_YES_PATTERNS):
        return "accept"
    return "neutral"


# ---------------------------------------------------------------------------
# Lightweight language detection (per-turn, so voice can track a merchant
# switching mid-conversation, per open challenge #4)
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_ROMAN_HINDI_WORDS = {
    "hai", "haan", "nahi", "nahin", "kya", "kaise", "aap", "aapka", "aapki",
    "kar", "karo", "kijiye", "chalo", "bhai", "theek", "accha", "shukriya",
    "dhanyavaad", "mera", "meri", "mujhe", "hum", "humein", "sakte", "sakta",
}


def detect_turn_language(body: str) -> Literal["hi", "en", "hi-en"]:
    has_devanagari = bool(_DEVANAGARI_RE.search(body))
    words = set(re.findall(r"[a-zA-Z]+", body.lower()))
    has_roman_hindi = bool(words & _ROMAN_HINDI_WORDS)
    if has_devanagari and not words:
        return "hi"
    if has_devanagari or has_roman_hindi:
        return "hi-en"
    return "en"


class ConversationStore:
    """Keeps ConversationState per conversation_id across the test window."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def get_or_create(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: str | None = None,
        trigger_id: str | None = None,
    ) -> ConversationState:
        if conversation_id not in self._states:
            self._states[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_id=trigger_id,
            )
        return self._states[conversation_id]

    def get(self, conversation_id: str) -> ConversationState | None:
        return self._states.get(conversation_id)

    def clear(self) -> None:
        self._states.clear()
