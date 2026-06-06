#!/usr/bin/env python3
"""
Offline evaluation harness.

Ranks the labelled seed candidates with the production model and reports the
same metrics the organisers use (NDCG@10, NDCG@50, MAP, P@10, composite), plus
two sanity checks: honeypot leakage and the title mix of the predicted top.

    python eval/run_eval.py --candidates ./candidates.jsonl \
        --labels eval/seed_labels.jsonl

Use it as a regression guard whenever you change weights or scoring logic.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker import features as F, scoring          # noqa: E402
from redrob_ranker.io_utils import stream_candidates       # noqa: E402
import metrics                                             # noqa: E402  (same dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", default="eval/seed_labels.jsonl")
    args = ap.parse_args()

    labels = {}
    with open(args.labels) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                labels[r["candidate_id"]] = r
    want = set(labels)

    scored = []
    for c in stream_candidates(args.candidates):
        cid = c.get("candidate_id")
        if cid in want:
            feats = F.extract(c)
            info = scoring.score_candidate(c, feats, scoring.DEFAULT_WEIGHTS)
            scored.append((info["score"], cid, info, labels[cid]))
            if len(scored) == len(want):
                break

    scored.sort(key=lambda x: (-x[0], x[1]))
    rels = [lab["relevance"] for _, _, _, lab in scored]

    m = metrics.composite(rels)
    print(f"Evaluated {len(scored)} labelled candidates.\n")
    print(f"  NDCG@10 : {m['ndcg@10']:.4f}")
    print(f"  NDCG@50 : {m['ndcg@50']:.4f}")
    print(f"  MAP     : {m['map']:.4f}")
    print(f"  P@10    : {m['p@10']:.4f}")
    print(f"  -----------------------------")
    print(f"  COMPOSITE: {m['composite']:.4f}   (0.5*N@10 + 0.3*N@50 + 0.15*MAP + 0.05*P@10)\n")

    # sanity: honeypot leakage into the predicted top 10
    hp_top10 = sum(1 for _, _, info, _ in scored[:10] if info["is_honeypot"])
    print(f"  Honeypots in predicted top-10 of labelled set: {hp_top10}")
    print("\n  Predicted order (top 12):")
    for i, (s, cid, info, lab) in enumerate(scored[:12], 1):
        flag = " [HONEYPOT]" if info["is_honeypot"] else ""
        print(f"   {i:2d}. rel={lab['relevance']}  score={s:.3f}  {lab['title']}{flag}")


if __name__ == "__main__":
    main()
