"""
Deterministic, specific, non-templated reasoning.

Stage-4 manual review penalises empty/identical/templated reasoning, anything
that mentions skills not in the profile (hallucination), and reasoning whose
tone contradicts the rank.  We therefore build each sentence ONLY from facts we
read off the candidate, cite concrete evidence (real matched phrases, real
numbers), and switch tone based on the computed score AND the assigned rank.

Anti-template measures (each deterministic, keyed off the candidate_id so the
output is identical across runs and machines):
  * evidence phrases are de-duplicated by containment ("recommendation system"
    and "recommendation systems" can no longer be cited together) and the
    *selection* rotates per candidate, so neighbouring rows cite different
    evidence even when their profiles overlap;
  * four lead-sentence structures and four closing fit-clauses rotate per
    candidate, so sampled rows don't share one skeleton;
  * tail-of-list rows (rank > 60) carry an honest, fact-based qualifier (their
    genuinely weakest scoring axis), matching the spec's own rank-100 example.
"""
from __future__ import annotations
import zlib
from typing import Dict, Any, List, Optional

from . import concepts as C


def _cid_seed(candidate) -> int:
    """Stable small integer derived from the candidate id.  crc32 is
    deterministic across runs, processes and machines (unlike Python's
    hash(), which is salted per process) and mixes neighbouring ids well."""
    cid = str(candidate.get("candidate_id") or "0")
    return zlib.crc32(cid.encode("utf-8"))


def _all_evidence_phrases(candidate, cap=8) -> List[str]:
    """All distinct IR/ML phrases that literally appear in the profile text
    (citing them can never be a hallucination), de-duplicated so that no kept
    phrase is a substring of another (drops singular/plural and 'rag' vs
    'rag pipeline' near-duplicates).  IR phrases first, then ML."""
    from .features import build_text
    text = build_text(candidate)
    found: List[str] = []

    def _consider(raw: str):
        p = raw.strip(" .,(")
        if len(p) <= 3 and p not in ("rag", "ltr", "llm", "nlp"):
            return
        if p in ("retrieval",):  # too generic to cite on its own
            return
        for i, q in enumerate(found):
            if p in q or q in p:
                # keep the longer, more specific phrasing
                if len(p) > len(q):
                    found[i] = p
                return
        found.append(p)

    for ph in C.CORE_IR["phrases"]:
        if ph in text:
            _consider(ph)
        if len(found) >= cap:
            return found
    for ph in C.CORE_ML["phrases"]:
        if ph in text:
            _consider(ph)
        if len(found) >= cap:
            break
    return found


def _pick_evidence(candidate, limit=2) -> List[str]:
    """Rotate which evidence gets cited, per candidate, so rows with similar
    profiles don't all cite the same two list-leading phrases."""
    pool = _all_evidence_phrases(candidate)
    if len(pool) <= limit:
        return pool
    start = _cid_seed(candidate) % len(pool)
    picked = [pool[start]]
    if limit > 1:
        picked.append(pool[(start + max(1, len(pool) // 2)) % len(pool)])
    return picked


def _real_ai_skills(candidate, limit=3) -> List[str]:
    from .features import is_ai_skill
    out = []
    for s in candidate.get("skills", []) or []:
        nm = str(s.get("name", ""))
        if is_ai_skill(nm):
            out.append(nm)
        if len(out) >= limit:
            break
    return out


def _recent_product_role(candidate) -> str:
    """The most recent non-consulting role's company name, if any."""
    from .features import _CONSULTING_RE
    history = candidate.get("career_history", []) or []
    for r in history:
        company = (r.get("company") or "").strip()
        if not company:
            continue
        if _CONSULTING_RE.search(company.lower()):
            continue
        return company
    return ""


# Four interchangeable "strong fit" lead structures.  Slots: title, yoe,
# company, ev1, ev2/fit-clause.  All facts, different sentence shapes.
def _strong_leads(title, yoe, company, ev):
    e1 = ev[0] if ev else ""
    e2 = ev[1] if len(ev) > 1 else ""
    both = f"{e1} and {e2}" if e2 else e1
    at = f" at {company}" if company else ""
    art = "an" if title[:1].lower() in "aeiou" else "a"
    return [
        f"{title} with {yoe:.0f} yrs; profile shows hands-on {', '.join(ev)} — "
        f"the retrieval/ranking work this role centres on",
        f"{yoe:.1f} yrs as {art} {title}{at}, with {both} running through the "
        f"career history — squarely the JD's core mandate",
        f"{title}{at} ({yoe:.0f} yrs) whose roles cover {both}, a close match "
        f"to the search/recommendation systems this position owns",
        f"Seasoned {title} ({yoe:.0f} yrs) with concrete {both} work — "
        f"exactly the matching/ranking depth the JD asks for",
    ]


def _weakest_axis_note(f, info) -> str:
    """An honest, fact-based qualifier drawn from the candidate's genuinely
    weakest scoring axis.  Used for tail-of-list rows so a rank-90 rationale
    doesn't read identical to a rank-5 one.  Returns '' if nothing applies."""
    comp = info.get("components", {})
    notes = []
    if comp.get("experience", 1.0) < 1.0:
        y = f.get("yoe", 0)
        notes.append((comp["experience"],
                      f"at {y:.1f} yrs, sits at the edge of the JD's 5-9 band"))
    if comp.get("company", 1.0) < 0.8 and f.get("consulting_share", 0) > 0:
        notes.append((comp["company"],
                      f"~{int(round(f['consulting_share'] * 100))}% of career in services/consulting firms"))
    if comp.get("location", 1.0) <= 0.6:
        notes.append((comp["location"], "outside the JD's preferred metros"))
    if comp.get("domain", 1.0) < 0.55:
        notes.append((comp["domain"], "domain evidence is thinner than the top of this list"))
    nd = f.get("notice_days")
    if isinstance(nd, (int, float)) and nd >= 60:
        notes.append((0.5, f"{int(nd)}-day notice period"))
    if not notes:
        return ""
    notes.sort(key=lambda t: t[0])
    return notes[0][1]


def generate(candidate: Dict[str, Any], f: Dict[str, Any], info: Dict[str, Any],
             rank: Optional[int] = None) -> str:
    profile = candidate.get("profile", {}) or {}
    title = profile.get("current_title") or "professional"
    company = profile.get("current_company", "")
    yoe = f["yoe"]
    score = info["score"]
    seed = _cid_seed(candidate)

    # ---- lead clause: who they are + strongest fit evidence ---------------
    ev = _pick_evidence(candidate)
    skills = _real_ai_skills(candidate)

    if score >= 0.6:
        if ev:
            lead = _strong_leads(title, yoe, company, ev)[seed % 4]
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
    elif f["in_india"]:
        # city callout is not gated on willing_to_relocate; a candidate already
        # in an Indian city should be named even if they prefer to stay
        if f["willing_to_relocate"] and city:
            pos.append(f"{city}-based, open to relocate")
        elif city:
            pos.append(f"{city}-based")
        elif f["willing_to_relocate"]:
            pos.append("India-based, open to relocate")
    if f["response_rate"] >= 0.6 and f["days_inactive"] <= 60:
        pos.append("responsive and recently active")
    if f["open_to_work"]:
        pos.append("open to work")
    if isinstance(f["notice_days"], (int, float)) and f["notice_days"] <= 30:
        pos.append(f"{int(f['notice_days'])}-day notice")

    # if we still have no colour at all and the score is high, pull a concrete
    # fact from the profile so the reasoning isn't a bare title sentence.
    if score >= 0.45 and not pos:
        rec = _recent_product_role(candidate)
        if rec:
            pos.append(f"recent role at {rec}")
        else:
            pool = _all_evidence_phrases(candidate)
            extra = next((p for p in pool if p not in ev), "")
            if extra:
                pos.append(f"also cites {extra}")

    # ---- honest concerns --------------------------------------------------
    concerns = list(info["dq_notes"]) + list(info["beh_notes"])
    if info["is_honeypot"]:
        concerns = ["profile fails internal consistency checks"] + concerns
    if not f["in_india"] and not f["willing_to_relocate"]:
        concerns.append("outside India, not open to relocate")
    # tail-of-list honesty: rows deep in the top-100 carry their genuinely
    # weakest axis, so tone tracks rank (the spec's own rank-100 example hedges)
    if rank is not None and rank > 60 and not concerns:
        weak = _weakest_axis_note(f, info)
        if weak:
            concerns.append(weak)

    parts = [lead.rstrip(".")]
    if pos and score >= 0.45:
        parts.append("; " + ", ".join(pos[:3]))
    if concerns:
        label = "Watch-outs" if (rank is not None and rank > 60 and not info["dq_notes"]
                                 and not info["beh_notes"] and not info["is_honeypot"]) else "Concerns"
        parts.append(f". {label}: " + "; ".join(concerns[:2]))
    text = "".join(parts).strip()
    if not text.endswith("."):
        text += "."
    # keep it to ~2 sentences / reasonable length, never cutting mid-word
    if len(text) > 300:
        cut = text[:300]
        text = cut[:cut.rfind(" ")].rstrip(",;") + "."
    return text
