"""
Feature extraction.  Turns a raw candidate dict into a flat, cheap-to-compute
feature record used by scoring and reasoning.  Pure Python, no model downloads,
so it is fully reproducible on CPU with no network.
"""
from __future__ import annotations
import datetime
from typing import Dict, Any

from . import concepts as C

DATASET_DATE = datetime.date(2026, 6, 1)


def _pdate(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _count_phrases(text: str, phrases) -> int:
    """Total (capped) occurrences of any phrase in text."""
    total = 0
    for ph in phrases:
        if ph in text:
            total += min(text.count(ph), 3)  # diminishing returns per phrase
    return total


def _distinct_phrases(text: str, phrases) -> int:
    return sum(1 for ph in phrases if ph in text)


def build_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) or {}
    parts = [
        profile.get("headline", ""), profile.get("summary", ""),
        profile.get("current_title", ""), profile.get("current_industry", ""),
    ]
    for role in candidate.get("career_history", []) or []:
        parts += [role.get("title", ""), role.get("description", ""),
                  role.get("industry", ""), role.get("company", "")]
    for s in candidate.get("skills", []) or []:
        parts.append(s.get("name", ""))
    for e in candidate.get("education", []) or []:
        parts += [e.get("field_of_study", ""), e.get("degree", "")]
    # pad with spaces so word-boundary-ish phrases like " nlp " can match
    return "  " + "  ".join(p for p in parts if p).lower() + "  "


def title_tier(title: str) -> str:
    t = (title or "").lower()
    for kw in C.TITLE_TIER_A:
        if kw in t:
            return "A"
    for kw in C.TITLE_TIER_D:
        if kw in t:
            return "D"
    for kw in C.TITLE_TIER_B:
        if kw in t:
            return "B"
    for kw in C.TITLE_TIER_C:
        if kw in t:
            return "C"
    return "C"  # unknown engineering-ish default


def extract(candidate: Dict[str, Any]) -> Dict[str, Any]:
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    text = build_text(candidate)

    # ---- domain / concept signal (the "understands the work" axis) --------
    f = {}
    f["ir_hits"] = _count_phrases(text, C.CORE_IR["phrases"])
    f["ir_distinct"] = _distinct_phrases(text, C.CORE_IR["phrases"])
    f["ml_hits"] = _count_phrases(text, C.CORE_ML["phrases"])
    f["ml_distinct"] = _distinct_phrases(text, C.CORE_ML["phrases"])
    f["data_hits"] = _count_phrases(text, C.DATA_ENG["phrases"])
    f["craft_hits"] = _count_phrases(text, C.ENG_CRAFT["phrases"])
    f["offdomain_hits"] = _count_phrases(text, C.OFF_DOMAIN["phrases"])
    f["framework_hits"] = _count_phrases(text, C.FRAMEWORK_FLUFF["phrases"])
    f["research_hits"] = _count_phrases(text, C.RESEARCH_ONLY["phrases"])

    # ---- title / role -----------------------------------------------------
    f["current_title"] = profile.get("current_title", "")
    f["title_tier"] = title_tier(profile.get("current_title", ""))
    ct = f["current_title"].lower()
    f["is_junior"] = any(k in ct for k in ("junior", "intern", "trainee", "associate", "fresher"))
    # best title across the whole career (a current manager may have been an MLE)
    tiers = [title_tier(r.get("title", "")) for r in history] + [f["title_tier"]]
    f["best_title_tier"] = min(tiers) if tiers else "C"  # A < B < C < D ordering by string

    # ---- experience -------------------------------------------------------
    f["yoe"] = float(profile.get("years_of_experience") or 0)

    # ---- company: product vs services ------------------------------------
    companies = [(r.get("company", "") or "").lower() for r in history]
    industries = [(r.get("industry", "") or "").lower() for r in history]
    cur_ind = (profile.get("current_industry", "") or "").lower()
    n_roles = max(1, len(history))
    consulting_roles = sum(
        1 for c, ind in zip(companies, industries)
        if any(k in c for k in C.CONSULTING_FIRMS) or any(k in ind for k in C.SERVICES_INDUSTRIES)
    )
    f["consulting_share"] = consulting_roles / n_roles
    f["all_services"] = (consulting_roles == len(history) and len(history) > 0) or \
                        (any(k in cur_ind for k in C.SERVICES_INDUSTRIES) and consulting_roles >= n_roles - 0)
    f["has_product_role"] = consulting_roles < len(history)

    # ---- location ---------------------------------------------------------
    loc = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    f["location_city"] = (profile.get("location", "") or "").split(",")[0].strip()
    f["in_india"] = ("india" in country) or any(
        city in loc for city in (C.PREFERRED_CITIES + C.WELCOME_CITIES + C.OTHER_INDIA_HINT))
    f["preferred_city"] = any(city in loc for city in C.PREFERRED_CITIES)
    f["welcome_city"] = any(city in loc for city in C.WELCOME_CITIES)
    f["willing_to_relocate"] = bool(signals.get("willing_to_relocate"))

    # ---- skills corroboration (anti keyword-stuffing) ---------------------
    ai_skill_terms = ("machine learning", "deep learning", "nlp", "ml", "llm",
                      "retrieval", "ranking", "recommendation", "embedding",
                      "transformer", "pytorch", "tensorflow", "search")
    ai_skills = [s for s in skills
                 if any(t in str(s.get("name", "")).lower() for t in ai_skill_terms)]
    f["n_ai_skills"] = len(ai_skills)
    # corroboration: endorsements + months of use + assessment scores
    corro = 0.0
    assess = signals.get("skill_assessment_scores", {}) or {}
    assess_lc = {str(k).lower(): v for k, v in assess.items()}
    for s in ai_skills:
        nm = str(s.get("name", "")).lower()
        used = (s.get("duration_months", 0) or 0) >= 12
        endorsed = (s.get("endorsements", 0) or 0) >= 5
        assessed = assess_lc.get(nm, 0) >= 60
        corro += (0.4 * used + 0.3 * endorsed + 0.3 * assessed)
    f["ai_skill_corroboration"] = corro / max(1, len(ai_skills)) if ai_skills else 0.0

    # recency of AI skills only (the "recent LangChain only" tell)
    ai_durations = [s.get("duration_months", 0) or 0 for s in ai_skills]
    f["ai_all_recent"] = bool(ai_durations) and max(ai_durations) < 12

    # ---- education --------------------------------------------------------
    edu = candidate.get("education", []) or []
    tiers_edu = [e.get("tier") for e in edu]
    f["edu_tier1"] = "tier_1" in tiers_edu
    f["edu_tier2"] = "tier_2" in tiers_edu

    # ---- behaviour / availability ----------------------------------------
    la = _pdate(signals.get("last_active_date"))
    f["days_inactive"] = (DATASET_DATE - la).days if la else 999
    f["response_rate"] = float(signals.get("recruiter_response_rate") or 0)
    f["open_to_work"] = bool(signals.get("open_to_work_flag"))
    f["interview_completion"] = float(signals.get("interview_completion_rate") or 0)
    f["notice_days"] = signals.get("notice_period_days")
    f["profile_completeness"] = float(signals.get("profile_completeness_score") or 0)
    f["saved_by_recruiters"] = int(signals.get("saved_by_recruiters_30d") or 0)
    f["github"] = float(signals.get("github_activity_score") if signals.get(
        "github_activity_score") is not None else -1)

    # ---- career stability (title-chaser tell) -----------------------------
    short_stints = sum(1 for r in history
                       if isinstance(r.get("duration_months"), int)
                       and 0 < r["duration_months"] < 20 and not r.get("is_current"))
    f["short_stints"] = short_stints

    # ---- "no recent hands-on code" tell -----------------------------------
    managerial_now = f["title_tier"] == "D" and any(
        k in f["current_title"].lower() for k in ("manager", "lead", "architect", "head", "director"))
    f["managerial_now"] = managerial_now

    return f
