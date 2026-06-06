"""
Honeypot / internal-consistency detection.

The dataset seeds ~80 "subtly impossible" profiles that the ground truth forces
to relevance tier 0.  Ranking any of them in the top 100 hurts the score, and
honeypot rate > 10% in the top 100 is an automatic Stage-3 disqualification.

We do NOT special-case known IDs.  Instead we read each profile for internal
contradictions — exactly the inspection a careful recruiter would do.  Every
check is a genuine impossibility or a strong overclaim, so false positives on
real candidates are rare and acceptable.

Returns (is_hard_honeypot, penalty_multiplier, reasons).
"""
from __future__ import annotations
import datetime
from typing import Dict, Any, List, Tuple

# Reference "today" for the dataset (data last touched 2026-06).  Tenure that
# exceeds time elapsed since a start date is physically impossible.
DATASET_DATE = datetime.date(2026, 6, 1)


def _pdate(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _months_between(d1: datetime.date, d2: datetime.date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def detect(candidate: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    reasons: List[str] = []
    hard = False  # a genuine impossibility -> force to the bottom
    soft_penalty = 1.0  # accumulates for overclaims that are merely suspicious

    yoe = profile.get("years_of_experience") or 0

    # --- 1. Impossible tenure / reversed dates in career history -----------
    summed_months = 0
    for role in history:
        sd = _pdate(role.get("start_date"))
        ed = _pdate(role.get("end_date"))
        dur = role.get("duration_months")
        if isinstance(dur, int):
            summed_months += dur
        if sd and role.get("is_current") and isinstance(dur, int):
            elapsed = _months_between(sd, DATASET_DATE)
            if dur > elapsed + 2:
                hard = True
                reasons.append(
                    f"claims {dur}mo in current role but only {elapsed}mo elapsed since start")
        if sd and ed:
            if ed < sd:
                hard = True
                reasons.append("a role ends before it starts")
            else:
                span = _months_between(sd, ed)
                if isinstance(dur, int) and abs(dur - span) > 9:
                    soft_penalty *= 0.5
                    reasons.append("stated duration disagrees with its own dates")

    # --- 2. Summed tenure wildly exceeds stated years of experience --------
    if summed_months > (yoe * 12) + 24:
        soft_penalty *= 0.35
        reasons.append(
            f"career tenures sum to {summed_months}mo, far above {yoe:.0f}y of experience")

    # --- 3. "Expert" in skills used for ~0 months --------------------------
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months", 0) or 0) == 0
    )
    if expert_zero >= 4:
        hard = True
        reasons.append(f"{expert_zero} advanced/expert skills with 0 months of use")
    elif expert_zero >= 2:
        soft_penalty *= 0.5
        reasons.append(f"{expert_zero} advanced/expert skills with 0 months of use")

    # --- 4. Assessment scores for skills not on the profile (ghost skills) -
    skill_names = {str(s.get("name", "")).lower() for s in skills}
    assess = signals.get("skill_assessment_scores", {}) or {}
    ghosts = [k for k in assess if str(k).lower() not in skill_names]
    if len(ghosts) >= 3:
        soft_penalty *= 0.5
        reasons.append(f"{len(ghosts)} assessment scores for skills not listed")

    # --- 5. Experience that predates plausible working age -----------------
    edu = candidate.get("education", []) or []
    start_years = [e.get("start_year") for e in edu if isinstance(e.get("start_year"), int)]
    if start_years and yoe:
        first_edu = min(start_years)
        # Working-years implied to begin before age ~17 is implausible.
        implied_career_start = DATASET_DATE.year - yoe
        if implied_career_start < first_edu - 1:
            soft_penalty *= 0.6
            reasons.append("years of experience predate the start of education")

    is_hard = hard or soft_penalty <= 0.2
    return is_hard, (0.01 if is_hard else soft_penalty), reasons
