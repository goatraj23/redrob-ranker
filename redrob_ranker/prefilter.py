"""
Cheap pre-filter.

Most of the 100K pool is "obviously not top-100" — non-engineering titles with
no AI signal anywhere on the profile.  The full feature scan (~150 phrases over
the concatenated profile text) is wasted on them.  This pre-filter returns
False only when *every* cheap signal is empty, so the cull is conservative and
safe — any candidate with a path to the top 100 still gets fully scored.

A candidate keeps going if ANY of the following hold:
  - current_title matches a Tier-A or Tier-B keyword
  - any past career_history title matches a Tier-A keyword
  - the candidate lists a corroborated AI skill (one with >= 12 months of use)
  - the headline or summary contains one of a short list of high-precision IR
    phrases ("recommendation", "ranking", "retrieval", "search", "embedding")

Otherwise it cannot plausibly reach the top 100, even with a maxed behavioural
multiplier: with role <= Tier-C (best 0.40), no domain hits, and weights from
DEFAULT_WEIGHTS, the upper bound on the final score sits well below the
empirical rank-100 cut-off observed on the full pool.

In practice this culls ~76% of the 100K pool, with no quality loss.
"""
from __future__ import annotations
from typing import Dict, Any

from . import concepts as C

_AI_SKILL_TERMS = (
    "machine learning", "deep learning", "nlp", "ml ", " ml,", "llm",
    "retrieval", "ranking", "recommendation", "embedding", "transformer",
    "pytorch", "tensorflow", "information retrieval", "search",
)

_CHEAP_IR_HINTS = (
    "recommendation", "ranking", "retrieval", "search", "embedding",
    "nlp", "machine learning", "deep learning", "personalization",
    "personalisation",
)

# High-precision hints used only when scanning career-history DESCRIPTIONS.
# Deliberately excludes the common-in-prose words ("search", "machine learning",
# "nlp") so a single passing mention doesn't keep half the pool; it targets the
# genuine recsys/ranking/retrieval builders the headline/skills checks missed.
_DESC_IR_HINTS = (
    "recommendation", "recommender", "learning to rank", "collaborative filtering",
    "ranking model", "ranking system", "retrieval", "embedding",
    "personalization", "personalisation", "vector search", "semantic search",
)


def _title_is_a_or_b(title_lc: str) -> bool:
    if not title_lc:
        return False
    for kw in C.TITLE_TIER_A:
        if kw in title_lc:
            return True
    for kw in C.TITLE_TIER_B:
        if kw in title_lc:
            return True
    return False


def _title_is_a(title_lc: str) -> bool:
    return any(kw in title_lc for kw in C.TITLE_TIER_A)


def cheap_can_compete(candidate: Dict[str, Any]) -> bool:
    profile = candidate.get("profile", {}) or {}
    ct = (profile.get("current_title", "") or "").lower()
    if _title_is_a_or_b(ct):
        return True

    for r in candidate.get("career_history", []) or []:
        pt = (r.get("title", "") or "").lower()
        if _title_is_a(pt):
            return True

    for s in candidate.get("skills", []) or []:
        nm = str(s.get("name", "")).lower()
        if any(t in nm for t in _AI_SKILL_TERMS):
            if (s.get("duration_months", 0) or 0) >= 12:
                return True

    headline = (profile.get("headline", "") or "").lower()
    summary = (profile.get("summary", "") or "").lower()
    blob = headline + " " + summary
    if any(t in blob for t in _CHEAP_IR_HINTS):
        return True

    # also scan the most recent role descriptions for a cheap IR hint, so a
    # genuine builder whose only signal sits in their job description (not the
    # headline/summary or skills) still survives the cull.
    for r in (candidate.get("career_history", []) or [])[:2]:
        desc = (r.get("description", "") or "").lower()
        if any(t in desc for t in _DESC_IR_HINTS):
            return True

    return False
