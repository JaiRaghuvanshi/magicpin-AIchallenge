"""
Generates submission.jsonl per challenge-brief.md §7.2 — one compose() call
per (merchant, trigger[, customer]) pair in dataset/test_pairs.json.

Uses the LLM provider if configured (ANTHROPIC_API_KEY / VERA_LLM_PROVIDER),
otherwise falls back to the deterministic rule-based composer automatically
(see vera/composer.py). Re-run after setting your API key to regenerate with
real LLM compositions.
"""
from __future__ import annotations

import json
from pathlib import Path

from vera.composer import compose
from test_local import load_all, DATASET

OUT_PATH = Path(__file__).parent / "submission.jsonl"


def main() -> None:
    categories = load_all("categories")
    merchants = load_all("merchants")
    triggers = load_all("triggers")
    customers = load_all("customers")
    pairs = json.loads((DATASET / "test_pairs.json").read_text())["pairs"]

    lines = []
    for pair in pairs:
        trigger = triggers[pair["trigger_id"]]
        merchant = merchants[pair["merchant_id"]]
        category = categories[merchant["category_slug"]]
        customer = customers.get(pair["customer_id"]) if pair["customer_id"] else None

        result = compose(category, merchant, trigger, customer)
        lines.append({
            "test_id": pair["test_id"],
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines)} lines to {OUT_PATH}")


if __name__ == "__main__":
    main()
