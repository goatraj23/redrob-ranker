"""
Redrob ranker — sample-only sandbox (Streamlit).

Satisfies the hackathon's mandatory sandbox requirement: accepts a SMALL sample
(<=100 candidates), runs the exact production ranker end-to-end on CPU with no
network, and produces the ranked CSV. It does NOT load the full pool.

Run locally:   streamlit run sandbox/app.py
Deploy free:   Streamlit Community Cloud or a HuggingFace Space.
"""
from __future__ import annotations
import io
import json
import os
import sys

import streamlit as st

# make the redrob_ranker package importable when run from anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from redrob_ranker import features as F, scoring, reasoning   # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_candidates.json")
MAX_CANDS = 100

st.set_page_config(page_title="Redrob Ranker — sandbox", layout="wide")
st.title("Redrob Candidate Ranker — sample sandbox")
st.caption("Ranks candidates for the *Senior AI Engineer (Founding Team)* JD. "
           "CPU only, no network, no LLM calls. Upload ≤100 candidates or use the bundled sample.")


def load_candidates(raw: bytes):
    text = raw.decode("utf-8").strip()
    # accept either a JSON array or JSONL
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data[:MAX_CANDS]


up = st.file_uploader("Candidate file (.json array or .jsonl, ≤100 records)",
                      type=["json", "jsonl"])
if up is not None:
    candidates = load_candidates(up.read())
    src = up.name
else:
    with open(SAMPLE) as fh:
        candidates = json.load(fh)
    src = "bundled sample_candidates.json (synthetic: strong fit, keyword-stuffer, honeypot, off-domain, consulting-only, inactive star)"

st.write(f"**Loaded {len(candidates)} candidates** from {src}.")

if st.button("Rank candidates", type="primary"):
    import csv as _csv
    scored = []
    for c in candidates:
        feats = F.extract(c)
        info = scoring.score_candidate(c, feats, scoring.DEFAULT_WEIGHTS)
        scored.append((info["score"], c, feats, info))
    gmax = max((s for s, *_ in scored), default=1.0) or 1.0
    rows = [(round(min(1.0, s / gmax), 4), c.get("candidate_id"), c, f, i)
            for s, c, f, i in scored]
    rows.sort(key=lambda r: (-r[0], r[1]))  # spec-valid: score desc, then id asc

    table = []
    out = io.StringIO()
    w = _csv.writer(out)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (printed, cid, c, f, info) in enumerate(rows, 1):
        why = reasoning.generate(c, f, info)
        w.writerow([cid, rank, f"{printed:.4f}", why])
        table.append({
            "rank": rank, "candidate_id": cid,
            "title": c.get("profile", {}).get("current_title", ""),
            "score": printed,
            "honeypot": "yes" if info["is_honeypot"] else "",
            "reasoning": why,
        })

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download ranked CSV", out.getvalue(),
                       file_name="submission_sample.csv", mime="text/csv")
    n_hp = sum(1 for r in table if r["honeypot"])
    st.success(f"Ranked {len(table)} candidates. Honeypots flagged & sent to the bottom: {n_hp}.")
