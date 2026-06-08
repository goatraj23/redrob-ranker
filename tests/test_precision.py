"""
Regression tests for the v2.1 precision/recall fixes.

Run with either:
    python -m pytest tests/ -q
    python tests/test_precision.py
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from redrob_ranker import features as F   # noqa: E402


def _cand(skills, desc, title="Software Engineer"):
    return {
        "candidate_id": "CAND_9999999",
        "profile": {"current_title": title, "headline": "", "summary": "",
                    "years_of_experience": 6, "location": "Pune, Maharashtra",
                    "country": "India", "current_industry": "Internet"},
        "career_history": [{"title": title, "company": "Acme", "industry": "Internet",
                            "description": desc, "duration_months": 40, "is_current": True,
                            "start_date": "2021-01-01", "end_date": None}],
        "education": [],
        "skills": [{"name": s, "proficiency": "advanced", "endorsements": 12,
                    "duration_months": 30} for s in skills],
        "redrob_signals": {},
    }


def test_is_ai_skill_word_boundaries():
    # the substring bug: HTML/XML contain "ml", Research contains "search"
    assert F.is_ai_skill("ML") is True
    assert F.is_ai_skill("Machine Learning") is True
    assert F.is_ai_skill("PyTorch") is True
    assert F.is_ai_skill("Search Relevance") is True
    assert F.is_ai_skill("HTML") is False
    assert F.is_ai_skill("XML") is False
    assert F.is_ai_skill("YAML") is False
    assert F.is_ai_skill("Research") is False
    assert F.is_ai_skill("Camel") is False


def test_substring_false_positives_gone():
    # common resume words that USED to trip "rag"/"ltr"/"api"/"acl"
    c = _cand(["HTML", "XML", "Research", "jQuery"],
              "Leveraged storage on Oracle with ultra-rapid pipelines; capital markets domain; "
              "computed the average latency across services.")
    f = F.extract(c)
    assert f["ir_distinct"] == 0, f"spurious IR hits: {f['ir_distinct']}"
    assert f["research_hits"] == 0, f"spurious research hits: {f['research_hits']}"
    assert f["n_ai_skills"] == 0, f"HTML/XML/Research wrongly counted: {f['n_ai_skills']}"


def test_expanded_lexicon_catches_paraphrases():
    # a real builder who never writes "RAG"/"Pinecone"
    c = _cand(["Python", "Spark"],
              "Designed a collaborative filtering recommender and a learning to rank model; "
              "ran offline NDCG evaluation and online A/B tests.")
    f = F.extract(c)
    assert f["ir_distinct"] >= 2, f"expected IR paraphrases, got {f['ir_distinct']}"
    assert f["ml_distinct"] >= 1


def test_real_rag_still_matches():
    c = _cand(["Python"], "Built a RAG pipeline with FAISS for document retrieval.")
    f = F.extract(c)
    assert f["ir_distinct"] >= 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
