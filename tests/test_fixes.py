"""
Tests locking in the v3 fixes.  Run with either:
    python -m pytest tests/test_fixes.py -q
    python3 tests/test_fixes.py          # no pytest needed
"""
from __future__ import annotations
import csv
import gzip
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from redrob_ranker import features as F, scoring, honeypots, reasoning   # noqa: E402


def _mk(cid="CAND_0000001", title="Machine Learning Engineer", company="Acme",
        start="2024-01-05", end=None, months=18, yoe=6.5, summary="",
        skills=None, history_extra=None):
    history = [{"company": company, "title": title, "start_date": start,
                "end_date": end, "duration_months": months, "is_current": end is None,
                "industry": "Software", "company_size": "201-500",
                "description": "Built embeddings-based semantic search and a ranking system."}]
    if history_extra:
        history += history_extra
    return {
        "candidate_id": cid,
        "profile": {"anonymized_name": "T", "headline": "ML", "summary": summary,
                    "location": "Pune", "country": "India", "years_of_experience": yoe,
                    "current_title": title, "current_company": company,
                    "current_company_size": "201-500", "current_industry": "Software"},
        "career_history": history,
        "education": [], "skills": skills or [],
        "redrob_signals": {"last_active_date": "2026-05-20", "recruiter_response_rate": 0.8,
                           "open_to_work_flag": True, "interview_completion_rate": 0.9,
                           "notice_period_days": 30, "profile_completeness_score": 90,
                           "saved_by_recruiters_30d": 3, "willing_to_relocate": True,
                           "skill_assessment_scores": {}, "github_activity_score": 50,
                           "preferred_work_mode": "hybrid"},
    }


# ---- fix 1: company-age (world-knowledge) honeypots ------------------------

def test_company_age_honeypot_flagged():
    hp = _mk(company="Krutrim", start="2018-11-05", end="2022-02-17", months=40)
    is_hp, mult, reasons = honeypots.detect(hp)
    assert is_hp and mult <= 0.02
    assert any("founded" in r for r in reasons)


def test_company_age_legit_hire_not_flagged():
    ok = _mk(company="Krutrim", start="2024-03-01", months=26)
    is_hp, _, reasons = honeypots.detect(ok)
    assert not is_hp and not any("founded" in r for r in reasons)


def test_company_age_lookalike_name_not_flagged():
    # exact-name matching: 'Sarvam Textiles' is not Sarvam AI
    ok = _mk(company="Sarvam Textiles", start="2015-01-01", end="2020-01-01", months=60)
    _, _, reasons = honeypots.detect(ok)
    assert not any("founded" in r for r in reasons)


# ---- fix 2: reasoning de-dup, variation, tail honesty ----------------------

def test_evidence_never_cites_singular_and_plural():
    c = _mk(summary="Built a recommendation system; maintain recommendation systems at scale.")
    ev = reasoning._all_evidence_phrases(c)
    assert not any(a != b and (a in b or b in a) for a in ev for b in ev), ev


def test_lead_templates_vary_across_candidates():
    leads = set()
    for i in range(1, 9):
        c = _mk(cid=f"CAND_000000{i}",
                summary="Shipped semantic search, learning to rank and vector search systems.")
        info = scoring.score_candidate(c, F.extract(c))
        txt = reasoning.generate(c, F.extract(c), info, rank=5)
        leads.add(txt.split(";")[0].split("(")[0][:18])
    assert len(leads) >= 3, f"lead shapes did not vary: {leads}"


def test_tail_rank_carries_qualifier():
    c = _mk(yoe=5.2, summary="Shipped semantic search and vector search systems.")
    f = F.extract(c)
    info = scoring.score_candidate(c, f)
    tail = reasoning.generate(c, f, info, rank=85)
    head = reasoning.generate(c, f, info, rank=5)
    assert ("Watch-outs:" in tail) or ("Concerns:" in tail)
    assert "Watch-outs:" not in head


# ---- fix 5: chunk-boundary line can no longer be dropped --------------------

def test_no_line_dropped_at_chunk_boundary():
    import rank as rank_mod
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "even.jsonl")
        # 10 lines of exactly 100 bytes -> with 2 workers the boundary at byte
        # 500 falls exactly on the start of line 6 (the old code dropped it)
        with open(path, "w") as fh:
            for i in range(10):
                rec = {"candidate_id": f"CAND_{i:07d}", "pad": ""}
                line = json.dumps(rec)
                rec["pad"] = "x" * (99 - len(line))
                fh.write(json.dumps(rec) + "\n")
        assert os.path.getsize(path) == 1000
        chunks = rank_mod._chunk_offsets(path, 2)
        seen = 0
        for s, e in chunks:
            heap, _, scanned, _ = rank_mod._worker((path, s, e, 50, scoring.DEFAULT_WEIGHTS))
            seen += scanned
        assert seen == 10, f"workers saw {seen}/10 lines"


# ---- fix 3: gzipped input is accepted ---------------------------------------

def test_gzip_input_round_trip():
    from rank import run
    cands = [_mk(cid=f"CAND_{i:07d}",
                 summary="Built semantic search and a ranking system.") for i in range(1, 6)]
    with tempfile.TemporaryDirectory() as d:
        gz = os.path.join(d, "c.jsonl.gz")
        with gzip.open(gz, "wt") as fh:
            for c in cands:
                fh.write(json.dumps(c) + "\n")
        out = os.path.join(d, "sub.csv")
        run(gz, out, n_workers=1)
        rows = list(csv.reader(open(out)))
        assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
        assert len(rows) == 6  # header + 5


# ---- fix 6: consulting word-boundary + managerial detection -----------------

def test_consulting_match_is_word_aware():
    assert F._CONSULTING_RE.search("ltimindtree")
    assert F._CONSULTING_RE.search("tata consultancy services")
    assert not F._CONSULTING_RE.search("multiplier")          # 'lti' inside a word
    assert not F._CONSULTING_RE.search("ultimate software")   # ditto


def test_architect_with_weak_domain_is_penalized():
    arch = _mk(title="Solutions Architect")
    arch["career_history"][0]["description"] = "Owns enterprise architecture governance."
    arch["profile"]["summary"] = "Architecture leadership."
    f = F.extract(arch)
    assert f["managerial_now"]
    info = scoring.score_candidate(arch, f)
    assert any("non-engineering role" in n for n in info["dq_notes"])


def test_hands_on_lead_engineer_not_penalized():
    lead = _mk(title="Lead AI Engineer",
               summary="Hands-on semantic search, vector search, learning to rank.")
    f = F.extract(lead)
    assert not f["managerial_now"]   # tier-A title is exempt


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
