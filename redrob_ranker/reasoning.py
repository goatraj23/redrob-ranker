"""
Deterministic, specific, non-templated reasoning.

Stage-4 manual review penalises empty/identical/templated reasoning, anything
that mentions skills not in the profile (hallucination), and reasoning whose
tone contradicts the rank.  We therefore build each sentence ONLY from facts we
read off the candidate, cite concrete evidence (real matched phrases, real
numbers), and switch tone based on the computed score.
"""
from __future__ import annotations
from typing import Dict, Any, List

from . import concepts as C


def _evidence_phrases(candidate, limit=2) -> List[str]:
    """Return up to `limit` IR/ML phrases that literally appear in the profile
    text, so citing them can never be a hallucination."""
    from .features import build_text
    text = build_text(candidate)
    found = []
    for ph in C.CORE_IR["phrases"]:
        if ph in text and ph not in ("retrieval",):
            found.append(ph)
        if len(found) >= limit:
            return found
    for ph in C.CORE_ML["phrases"]:
        p = ph.strip(" .,(")
        if ph in text and len(p) > 3:
            found.append(p)
        if len(found) >= limit:
            break
    return found


def _real_ai_skills(candidate, limit=3) -> List[str]:
    out = []
    terms = ("machine learning", "deep learning", "nlp", "llm", "retrieval",
             "ranking", "recommendation", "embedding", "transformer",
             "pytorch", "tensorflow", "search", "information retrieval")
    for s in candidate.get("skills", []) or []:
        nm = str(s.get("name", ""))
        if any(t in nm.lower() for t in terms):
            out.append(nm)
        if len(out) >= limit:
            break
    return out


def generate(candidate: Dict[str, Any], f: Dict[str, Any], info: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) or {}
    title = profile.get("current_title", "professional")
    company = profile.get("current_company", "")
    yoe = f["yoe"]
    score = info["score"]

    # ---- lead clause: who they are + strongest fit evidence ---------------
    ev = _evidence_phrases(candidate)
    skills = _real_ai_skills(candidate)
    where = company and f"at {company}" or ""

    if score >= 0.6:
        if ev:
            lead = (f"{title} with {yoe:.0f} yrs; profile shows hands-on "
                    f"{', '.join(ev)} — the retrieval/ranking work the role centres on")
        elif skills:
            lead = (f"{title} with {yoe:.0f} yrs and corroborated {', '.join(skills[:2])} "
                    f"experience relevant to the matching/ranking mandate")
        else:
            lead = f"{title} with {yoe:.0f} yrs of applied engineering relevant to the role"
    elif score >= 0.4:
        if ev:
            lead = (f"{title} with {yoe:.0f} yrs; some relevant signal "
                    f"({', '.join(ev)}) but not a clear core-IR fit")
        else:
            lead = (f"{title} with {yoe:.0f} yrs; adjacent engineering background "
                    f"with limited retrieval/ranking evidence")
    else:
        lead = (f"{title} with {yoe:.0f} yrs; profile is off the core "
                f"retrieval/ranking profile the role needs")

    # ---- positive availability colour (only if genuinely good) ------------
    pos = []
    city = f.get("location_city") or ""
    if f["preferred_city"]:
        pos.append(f"{city}-based (preferred location)" if city else "Pune/Noida-based")
    elif f["welcome_city"]:
        pos.append(f"{city}-based" if city else "in a welcome metro")
    elif f["in_india"] and f["willing_to_relocate"]:
        pos.append(f"{city}-based, open to relocate" if city else "India-based, open to relocate")
    if f["response_rate"] >= 0.6 and f["days_inactive"] <= 60:
        pos.append("responsive and recently active")
    if f["open_to_work"]:
        pos.append("open to work")
    if isinstance(f["notice_days"], (int, float)) and f["notice_days"] <= 30:
        pos.append(f"{int(f['notice_days'])}-day notice")

    # ---- honest concerns --------------------------------------------------
    concerns = list(info["dq_notes"]) + list(info["beh_notes"])
    if info["is_honeypot"]:
        concerns = ["profile fails internal consistency checks"] + concerns
    if not f["in_india"] and not f["willing_to_relocate"]:
        concerns.append("outside India, not open to relocate")

    parts = [lead.rstrip(".")]
    if pos and score >= 0.45:
        parts.append("; " + ", ".join(pos[:3]))
    if concerns:
        parts.append(". Concerns: " + "; ".join(concerns[:2]))
    text = "".join(parts).strip()
    if not text.endswith("."):
        text += "."
    # keep it to ~2 sentences / reasonable length
    return text[:300]
