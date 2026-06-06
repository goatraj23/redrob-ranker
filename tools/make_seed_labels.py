#!/usr/bin/env python3
"""
Bootstrap a labelled validation set for offline evaluation.

Because the challenge has NO public leaderboard, you must measure quality
yourself.  This builds a stratified seed set of ~100 candidates spanning the
relevance spectrum, labelled by an INDEPENDENT, deliberately-simple rubric that
reads career-history *descriptions* (what the person actually did) rather than
skill tags — so it is not a copy of the ranker's scoring function and can catch
regressions.

These are bootstrap labels.  Replace / augment them with human review for the
highest-trust evaluation. No AI is used — the rubric is pure rules over fields.

    python tools/make_seed_labels.py --candidates ./candidates.jsonl \
        --out eval/seed_labels.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker import honeypots, features as F  # noqa: E402

STRONG_IR = ["recommendation system", "recommender", "learning to rank",
             "learning-to-rank", "information retrieval", "search relevance",
             "ranking model", "ranking system", "re-ranking", "semantic search",
             "vector search", "retrieval-augmented", " rag", "embeddings",
             "personalization", "personalisation", "nearest neighbor", "faiss",
             "elasticsearch", "two-tower", "candidate generation"]
STRONG_ML = ["machine learning", "deep learning", "natural language processing",
             " nlp", "transformer", "model training", "xgboost", "lightgbm",
             "pytorch", "tensorflow", "fine-tun", "classification model"]
OFF_DOM = ["computer vision", "image classification", "object detection",
           "opencv", "speech recognition", "asr", "robotics", "slam"]


def evidence_text(c):
    parts = [c.get("profile", {}).get("summary", "")]
    for r in c.get("career_history", []) or []:
        parts.append(r.get("description", ""))
        parts.append(r.get("title", ""))
    return ("  " + "  ".join(p for p in parts if p) + "  ").lower()


def rubric_label(c) -> int:
    is_hp, _, _ = honeypots.detect(c)
    if is_hp:
        return 0
    f = F.extract(c)
    ev = evidence_text(c)
    strong_ir = any(t in ev for t in STRONG_IR)
    strong_ml = any(t in ev for t in STRONG_ML)
    off = sum(t in ev for t in OFF_DOM) >= 2 and not strong_ir and " nlp" not in ev

    tt = f["title_tier"]
    bt = f["best_title_tier"]
    cls = min(tt, bt)  # 'A'<'B'<'C'<'D'

    title_l = (c.get("profile", {}).get("current_title", "") or "").lower()
    off_title = any(k in title_l for k in ("computer vision", "vision engineer",
                                           "speech", "robotics", "image"))

    if cls == "D":
        return 1 if strong_ir else 0
    if off or (off_title and not strong_ir):
        return 1

    s = {"A": 3, "B": 2, "C": 1}.get(cls, 0)
    if strong_ir:
        s += 1
    if f["all_services"]:
        s -= 1
    if not (4 <= f["yoe"] <= 10):
        s -= 1
    if cls == "A" and strong_ir and f["has_product_role"] and 5 <= f["yoe"] <= 9:
        s = 4
    if cls in ("B", "C") and not (strong_ir or strong_ml):
        s = min(s, 1)
    return max(0, min(4, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="eval/seed_labels.jsonl")
    ap.add_argument("--scan_cap", type=int, default=100000)
    args = ap.parse_args()

    targets = {4: 12, 3: 16, 2: 20, 1: 25, 0: 19}
    buckets = {k: [] for k in targets}
    hp_quota = 8
    hp_list = []
    from redrob_ranker.io_utils import stream_candidates

    n = 0
    for c in stream_candidates(args.candidates):
        n += 1
        if n > args.scan_cap:
            break
        is_hp, _, _ = honeypots.detect(c)
        lab = rubric_label(c)
        rec = {
            "candidate_id": c.get("candidate_id"),
            "relevance": lab,
            "title": c.get("profile", {}).get("current_title", ""),
            "is_honeypot": is_hp,
        }
        if is_hp:
            if len(hp_list) < hp_quota:
                hp_list.append(rec)
            continue
        if len(buckets[lab]) < targets[lab]:
            buckets[lab].append(rec)
        buckets_full = all(len(buckets[k]) >= targets[k] for k in targets)
        if buckets_full and len(hp_list) >= hp_quota:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # honeypots are tier-0 by definition; include them in the tier-0 group
    all_recs = hp_list + [r for k in sorted(targets) for r in buckets[k]]
    with open(args.out, "w") as fh:
        for r in all_recs:
            fh.write(json.dumps(r) + "\n")
    dist = {k: len(buckets[k]) for k in sorted(targets, reverse=True)}
    dist[0] += len(hp_list)
    print(f"Wrote {len(all_recs)} seed labels to {args.out}")
    print(f"Tier distribution (4..0): {dist}  | honeypots included: {len(hp_list)}")


if __name__ == "__main__":
    main()
