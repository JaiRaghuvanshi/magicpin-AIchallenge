# Vera Challenge Bot

An HTTP bot that composes and manages WhatsApp-style merchant/customer conversations
for the magicpin "Vera" challenge, implementing the 4-context composition contract
(`challenge-brief.md` §5) and the 5-endpoint judge harness (`challenge-testing-brief.md` §2).

## Architecture

- **`bot.py`** — FastAPI server exposing `/v1/context`, `/v1/tick`, `/v1/reply`,
  `/v1/healthz`, `/v1/metadata`, `/v1/teardown`.
- **`vera/composer.py`** — resolves each trigger into a small set of *verifiable*
  anchor facts (never lets the model see or invent beyond them), builds a
  category/voice-aware prompt, calls the LLM, then validates/repairs the output
  (taboo vocab, multi-CTA, repeats). Falls back to a deterministic rule-based
  composer if no LLM key is configured or the call fails, so the bot is never
  silently broken.
- **`vera/conversation.py`** + **`conversation_handlers.py`** — multi-turn state
  machine: auto-reply detection (lexical markers + 3x verbatim-repeat counter),
  intent-transition handling (skip re-qualification on an explicit "yes"), graceful
  exit after 3 unanswered nudges or a decline, lightweight per-turn language detection.
- **`vera/context_store.py`** — versioned, idempotent `(scope, context_id)` store
  per the `/v1/context` contract.
- **`vera/llm_providers.py`** — pluggable LLM interface; defaults to Claude, one env
  var away from OpenAI.

## Running locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # optional — falls back to rule-based composer without it
uvicorn bot:app --host 0.0.0.0 --port 8080
```

No network/FastAPI? Exercise the composer and conversation logic directly:
```bash
python3 test_local.py
```

Regenerate the 30-pair submission file:
```bash
python3 make_submission.py
```

Self-test against the harness once your bot is running and you have an LLM key
configured in `judge_simulator.py`'s CONFIGURATION section:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

## Deploying for submission

Any host that gives a public HTTPS URL works (Render, Railway, Fly.io, ngrok for
quick local tests). Render example:

1. Push this folder to a GitHub repo.
2. Render → New Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
5. Add env vars: `ANTHROPIC_API_KEY`, `VERA_TEAM_NAME`, `VERA_TEAM_MEMBERS`, `VERA_CONTACT_EMAIL`
6. Deploy → verify `curl https://<your-url>/v1/healthz` returns `{"status": "ok", ...}`
7. Submit that URL.

Free tiers may cold-start on the first request — ping `/v1/healthz` a minute before
the test window opens, or use an always-on instance, to avoid eating the 30s per-call
timeout on turn 1.

## Known tradeoffs

- Validation is regex/substring-based, not a second LLM critique pass — a deliberate
  latency/cost tradeoff given the 30s-per-call budget.
- `/v1/tick` suppression is process-lifetime ("never refire this suppression_key"),
  not time-windowed — fine for a 60-minute test, not for production.
- Not tested against a live LLM or over real HTTP in the environment this was built
  in (no outbound network there); `vera/` logic is verified directly via
  `test_local.py`, but run `uvicorn` + `judge_simulator.py` yourself before submitting.

## Files

| File | Purpose |
|---|---|
| `bot.py` | FastAPI server — the 5 required endpoints + `/v1/teardown` |
| `vera/composer.py` | Fact extraction, prompting, validation, rule-based fallback |
| `vera/conversation.py` | State, auto-reply/intent/language detection |
| `vera/conversation_handlers.py` | `respond()` — optional multi-turn deliverable |
| `vera/context_store.py` | Versioned idempotent context store |
| `vera/llm_providers.py` | Pluggable LLM interface (Anthropic default) |
| `test_local.py` | Direct logic tests, no network/FastAPI required |
| `make_submission.py` | Generates `submission.jsonl` |
| `submission.jsonl` | 30 canonical test pairs (rule-based fallback output — regenerate with an API key) |
| `dataset/` | Full 50/200/100 expanded dataset + original seeds (for `judge_simulator.py`) |
