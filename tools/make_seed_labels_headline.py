#!/usr/bin/env python3
"""
Bootstrap a SECOND, INDEPENDENT validation label set for cross-lens evaluation.

`tools/make_seed_labels.py` (v1) reads the *full* career-history descriptions —
the same lens the production ranker uses.  Cross-checking against labels built
the same way is a regression guard, not an out-of-distribution check.

This rubric deliberately uses a *different* lens: it scores candidates from the
profile **headline + summary + current_title only**, and from a heuristic
"product-company" prior (recent company name not in CONSULTING_FIRMS).  It does
NOT read career_history descriptions.  If the ranker still scores well against
labels built this way, it means the system generalises beyond the lens its own
features use, instead of just being self-consistent.

This is the validation that makes the "high NDCG" claim defensible.

    python3 tools/make_seed_labels_headline.py --candidates ./candidates.jsonl \
        --out eval/seed_labels_headline.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker import honeypots                  # noqa: E402
from redrob_ranker.concepts import (                  # noqa: E402
    TITLE_TIER_A, TITLE_TIER_B, TITLE_TIER_C, TITLE_TIER_D, CONSULTING_FIRMS,
)

HEADLINE_IR = (
    "recommendation", "recommender", "ranking", "retrieval", "search relevance",
    "semantic search", "vector search", "embedding", "personalization",
    "personalisation", "information retrieval",
)
HEADLINE_ML = (
    "machine learning", "deep learning", "natural language processing",
    "nlp", "transformer", "llm", "large language model", "ml",
)
HEADLINE_OFF = (
    "computer vision", "image classification", "object detection", "opencv",
    "speech recognition", "asr", "robotics", "slam",
)


def headline_text(c) -> str:
    p = c.get("profile", {}) or {}
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    return ("  " + "  ".join(s for s in parts if s) + "  ").lower()


def tier_from_title(title_lc: str) -> str:
    for kw in TITLE_TIER_A:
        if kw in title_lc:
            return "A"
    for kw in TITLE_TIER_D:
        if kw in title_lc:
            return "D"
    for kw in TITLE_TIER_B:
        if kw in title_lc:
            return "B"
    for kw in TITLE_TIER_C:
        if kw in title_lc:
            return "C"
    return "C"


def is_product_recent(c) -> bool:
    """True if the most-recent career-history company is NOT a known services firm."""
    for r in c.get("career_history", []) or []:
        comp = (r.get("company") or "").lower()
        if not comp:
            continue
        return not any(k in comp for k in CONSULTING_FIRMS)
    return True  # no history -> don't penalise here


def rubric_label(c) -> int:
    is_hp, _, _ = honeypots.detect(c)
    if is_hp:
        return 0

    p = c.get("profile", {}) or {}
    cur_title = (p.get("current_title", "") or "").lower()
    yoe = float(p.get("years_of_experience") or 0)
    ht = headline_text(c)

    tier = tier_from_title(cur_title)
    ir = sum(1 for t in HEADLINE_IR if t in ht)
    ml = sum(1 for t in HEADLINE_ML if t in ht)
    off = sum(1 for t in HEADLINE_OFF if t in ht)
    product = is_product_recent(c)

    # Tier-D current title: only redeem if there's at least one IR hit in the
    # headline; never a top-tier label.
    if tier == "D":
        return 1 if ir >= 1 else 0

    # Off-domain dominance kills the score unless IR is also present.
    if off >= 2 and ir == 0:
        return 1

    base = {"A": 3, "B": 2, "C": 1}.get(tier, 0)

    if ir >= 2:
        base += 1
    elif ir == 1:
        base += 0  # weakly positive but doesn't bump the tier
    elif ml >= 2 and tier != "A":
        base += 0
    elif ir == 0 and ml == 0:
        base -= 1  # no AI signal in summary at all

    if not (4 <= yoe <= 10):
        base -= 1

    if not product:
        base -= 1

    if tier == "A" and ir >= 2 and 5 <= yoe <= 9 and product:
        base = 4

    return max(0, min(4, base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="eval/seed_labels_headline.jsonl")
    ap.add_argument("--scan_cap", type=int, default=100000)
    args = ap.parse_args()

    targets = {4: 12, 3: 16, 2: 20, 1: 25, 0: 19}
    buckets = {k: [] for k in targets}
    hp_quota = 8
    hp_list = []

    from redrob_ranker.io_utils import stream_candidates

    n = 0
    for c in stream_candidates(args.candidates):
        n += 1
        if n > args.scan_cap:
            break
        is_hp, _, _ = honeypots.detect(c)
        lab = rubric_label(c)
        rec = {
            "candidate_id": c.get("candidate_id"),
            "relevance": lab,
            "title": c.get("profile", {}).get("current_title", ""),
            "is_honeypot": is_hp,
            "lens": "headline_only",
        }
        if is_hp:
            if len(hp_list) < hp_quota:
                hp_list.append(rec)
            continue
        if len(buckets[lab]) < targets[lab]:
            buckets[lab].append(rec)
        if all(len(buckets[k]) >= targets[k] for k in targets) and len(hp_list) >= hp_quota:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_recs = hp_list + [r for k in sorted(targets) for r in buckets[k]]
    with open(args.out, "w") as fh:
        for r in all_recs:
            fh.write(json.dumps(r) + "\n")
    dist = {k: len(buckets[k]) for k in sorted(targets, reverse=True)}
    dist[0] += len(hp_list)
    print(f"Wrote {len(all_recs)} headline-only seed labels to {args.out}")
    print(f"Tier distribution (4..0): {dist}  | honeypots included: {len(hp_list)}")


if __name__ == "__main__":
    main()
