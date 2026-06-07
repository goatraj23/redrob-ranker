"""
Composite scoring.

    final = base_fit(role, domain, experience, company, skills, education, location)
            x disqualifier_multiplier
            x behavioural_availability_multiplier
            x honeypot_multiplier

Base fit is a transparent weighted sum of bounded [0,1] component scores.  The
multipliers encode the JD's hard rules ("title-chaser", "consulting-only",
"inactive and unresponsive isn't actually available").  Every number is
defensible by reference to job_description.docx.

DEFAULT_WEIGHTS are deliberately set by hand to sensible priors so the system
works with zero training and is easy to defend.  Re-balance them here if you
want to change the model's emphasis; the ranker is fully deterministic.

The role share (0.22) is lowered relative to a naive "role first" weighting
because the JD explicitly allows a Tier-B candidate with strong career-history
domain evidence (a "Tier-5 recovery") to compete.  Experience decays steeply
outside the 5-9 yr band, and a junior-titled candidate is multiplied by 0.55,
so 3-yr / 15-yr / Junior outliers don't sneak into the top 100.
"""
from __future__ import annotations
import math
from typing import Dict, Any, Tuple

from . import concepts as C
from . import honeypots

DEFAULT_WEIGHTS = {
    "role": 0.22,
    "domain": 0.32,
    "experience": 0.14,
    "company": 0.12,
    "skills": 0.10,
    "education": 0.04,
    "location": 0.06,
}


def _sat(x: float, k: float) -> float:
    """Saturating 0..1 transform: 0 -> 0, grows, asymptotes to 1."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-x / k)


# ---- component scores (each returns 0..1) ---------------------------------

def role_score(f) -> float:
    base = {"A": 1.0, "B": 0.62, "C": 0.40, "D": 0.12}[f["title_tier"]]
    best = {"A": 1.0, "B": 0.62, "C": 0.40, "D": 0.12}[f["best_title_tier"]]
    # a current non-core title can be redeemed by a clearly core past role
    score = max(base, 0.55 * base + 0.45 * best)
    # seniority: a "Junior/Intern/Associate" core title is not a senior hire
    if f.get("is_junior"):
        score *= 0.55
    return score


def domain_score(f) -> float:
    # IR is the decisive axis; ML supports; data/craft are minor.
    ir = _sat(f["ir_hits"] * 1.0 + f["ir_distinct"] * 0.6, k=4.0)
    ml = _sat(f["ml_hits"] * 0.8 + f["ml_distinct"] * 0.5, k=5.0)
    data = _sat(f["data_hits"], k=4.0)
    craft = _sat(f["craft_hits"], k=6.0)
    score = 0.58 * ir + 0.30 * ml + 0.08 * data + 0.04 * craft
    return min(1.0, score)


def experience_score(f) -> float:
    y = f["yoe"]
    if C.EXP_IDEAL_LOW <= y <= C.EXP_IDEAL_HIGH:
        return 1.0
    if C.EXP_OK_LOW <= y <= C.EXP_OK_HIGH:
        return 0.9
    # steeper decay outside the OK band so a 3-yr or 15-yr candidate has to be
    # exceptional on every other axis to crack the top 100.
    if y < C.EXP_OK_LOW:
        return max(0.10, 1.0 - (C.EXP_OK_LOW - y) * 0.25)
    return max(0.15, 1.0 - (y - C.EXP_OK_HIGH) * 0.15)


def company_score(f) -> float:
    if f["all_services"]:
        return 0.25
    # reward having product-company exposure; penalise heavy services share
    return max(0.3, 1.0 - 0.7 * f["consulting_share"])


def skills_score(f) -> float:
    # corroborated AI skills, not raw keyword count
    breadth = _sat(f["n_ai_skills"], k=4.0)
    return min(1.0, 0.35 * breadth + 0.65 * f["ai_skill_corroboration"])


def education_score(f) -> float:
    if f["edu_tier1"]:
        return 1.0
    if f["edu_tier2"]:
        return 0.75
    return 0.5


def location_score(f) -> float:
    if f["preferred_city"]:
        return 1.0
    if f["welcome_city"]:
        return 0.9
    if f["in_india"]:
        return 0.75 if f["willing_to_relocate"] else 0.6
    # outside India: case-by-case, no visa sponsorship
    return 0.45 if f["willing_to_relocate"] else 0.3


# ---- multipliers ----------------------------------------------------------

def disqualifier_multiplier(f) -> Tuple[float, list]:
    m = 1.0
    notes = []
    # consulting-only entire career, with no product role
    if f["all_services"] and not f["has_product_role"]:
        m *= 0.40
        notes.append("entire career in IT-services/consulting")
    # off-domain dominant (CV/speech/robotics) without NLP/IR alongside
    if f["offdomain_hits"] >= 3 and f["ir_distinct"] == 0 and f["ml_distinct"] <= 1:
        m *= 0.45
        notes.append("primary expertise is vision/speech/robotics, not NLP/IR")
    # pure-research signal with no product company exposure
    if f["research_hits"] >= 3 and not f["has_product_role"] and f["craft_hits"] == 0:
        m *= 0.45
        notes.append("research-heavy profile without production deployment")
    # "recent LangChain only": framework is the only AI signal and all recent
    if (f["framework_hits"] >= 1 and f["ir_distinct"] == 0
            and f["ml_distinct"] <= 1 and f["ai_all_recent"]):
        m *= 0.55
        notes.append("AI experience is recent framework usage only")
    # title-chaser: several short non-current stints
    if f["short_stints"] >= 3:
        m *= 0.75
        notes.append("frequent <2yr job changes")
    # (the "manager/architect now with weak hands-on signal" penalty is applied
    #  in score_candidate, where the domain component is already computed.)
    return m, notes


def behavioural_multiplier(f) -> Tuple[float, list]:
    m = 1.0
    notes = []
    # activity recency
    if f["days_inactive"] > 150:
        m *= 0.70
        notes.append(f"inactive ~{f['days_inactive']//30} months")
    elif f["days_inactive"] > 75:
        m *= 0.88
    # responsiveness to recruiters
    rr = f["response_rate"]
    if rr < 0.15:
        m *= 0.72
        notes.append(f"low recruiter response rate ({rr:.0%})")
    elif rr < 0.35:
        m *= 0.90
    elif rr >= 0.7:
        m *= 1.05
    # availability intent
    if f["open_to_work"]:
        m *= 1.05
    # interview reliability
    if f["interview_completion"] and f["interview_completion"] < 0.5:
        m *= 0.9
    # notice period (JD wants <=30, can buy out 30)
    nd = f["notice_days"]
    if isinstance(nd, (int, float)):
        if nd <= 30:
            m *= 1.04
        elif nd >= 90:
            m *= 0.9
            notes.append(f"long notice period ({nd}d)")
    # demand signal
    if f["saved_by_recruiters"] >= 5:
        m *= 1.03
    return min(m, 1.18), notes


def score_candidate(candidate: Dict[str, Any], features: Dict[str, Any],
                    weights=DEFAULT_WEIGHTS) -> Dict[str, Any]:
    comp = {
        "role": role_score(features),
        "domain": domain_score(features),
        "experience": experience_score(features),
        "company": company_score(features),
        "skills": skills_score(features),
        "education": education_score(features),
        "location": location_score(features),
    }
    base = sum(weights[k] * comp[k] for k in weights)

    dq_mult, dq_notes = disqualifier_multiplier(features)
    # apply the "no recent code" manager penalty now that domain is known
    if features["managerial_now"] and comp["domain"] < 0.35:
        dq_mult *= 0.8
        dq_notes.append("currently in a non-engineering role with weak hands-on signal")

    beh_mult, beh_notes = behavioural_multiplier(features)
    is_hp, hp_mult, hp_reasons = honeypots.detect(candidate)

    # Internal score is intentionally NOT clamped to 1.0: the behavioural
    # multiplier (up to ~1.18) must be able to separate otherwise-tied strong
    # candidates so the top-10 ordering stays discriminative.  rank.py
    # normalises to [0,1] for display.
    final = max(0.0, base * dq_mult * beh_mult * hp_mult)

    return {
        "score": final,
        "base": base,
        "components": comp,
        "dq_mult": dq_mult,
        "dq_notes": dq_notes,
        "beh_mult": beh_mult,
        "beh_notes": beh_notes,
        "is_honeypot": is_hp,
        "hp_mult": hp_mult,
        "hp_reasons": hp_reasons,
    }
