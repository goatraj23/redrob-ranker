"""
Fast, dependency-free tests. Run with either:
    python -m pytest tests/ -q
    python3 tests/test_basic.py      # no pytest needed
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from redrob_ranker import features as F, scoring, honeypots   # noqa: E402

SAMPLE = os.path.join(ROOT, "sandbox", "sample_candidates.json")
with open(SAMPLE) as fh:
    SAMPLES = {c["candidate_id"]: c for c in json.load(fh)}


def _score(cid):
    c = SAMPLES[cid]
    return scoring.score_candidate(c, F.extract(c), scoring.DEFAULT_WEIGHTS)


def test_honeypot_is_flagged():
    is_hp, mult, reasons = honeypots.detect(SAMPLES["CAND_9000003"])
    assert is_hp, "impossible-tenure profile must be flagged"
    assert mult <= 0.02 and reasons


def test_honeypot_pushed_to_bottom():
    assert _score("CAND_9000003")["score"] < 0.1


def test_strong_fit_beats_keyword_stuffer():
    assert _score("CAND_9000001")["score"] > _score("CAND_9000002")["score"]


def test_keyword_stuffer_scores_low():
    # Marketing Manager stuffed with AI skills should not look like a fit
    assert _score("CAND_9000002")["score"] < 0.4


def test_offdomain_penalized():
    assert _score("CAND_9000001")["score"] > _score("CAND_9000004")["score"]


def test_consulting_only_penalized():
    info = _score("CAND_9000005")
    assert info["dq_mult"] < 1.0


def test_inactive_star_downweighted():
    # great on paper, but inactive + 5% response rate -> below the active star
    assert _score("CAND_9000006")["score"] < _score("CAND_9000001")["score"]
    assert _score("CAND_9000006")["beh_mult"] < 0.8


def test_rank_output_is_spec_shaped():
    from rank import run
    # the streaming loader expects JSONL, so write the samples as JSONL first
    with tempfile.TemporaryDirectory() as d:
        jl = os.path.join(d, "c.jsonl")
        with open(jl, "w") as fh:
            for c in SAMPLES.values():
                fh.write(json.dumps(c) + "\n")
        out = os.path.join(d, "sub.csv")
        # single worker keeps the test stable across machines (so the 6-line
        # fixture isn't split across processes)
        run(jl, out, n_workers=1)
        import csv
        rows = list(csv.reader(open(out)))
        assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
        body = rows[1:]
        ranks = [int(r[1]) for r in body]
        scores = [float(r[2]) for r in body]
        assert ranks == list(range(1, len(body) + 1)), "ranks must be 1..N unique"
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "non-increasing"
        # honeypot must not be ranked #1
        assert body[0][0] != "CAND_9000003"


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
