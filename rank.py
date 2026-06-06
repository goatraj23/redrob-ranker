#!/usr/bin/env python3
"""
Redrob ranker — entrypoint.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Streams the candidate pool, scores each candidate with a transparent composite
model (role + domain + experience + company + skills + education + location,
modified by disqualifier / behavioural / honeypot multipliers), and writes the
top-100 CSV in the exact submission format.

Compute profile: CPU only, no network, single pass, bounded memory (keeps only
a small heap of the current best candidates).  Runs on the full 100K pool in a
few seconds.
"""
from __future__ import annotations
import argparse
import csv
import heapq
import json
import sys
import time

from redrob_ranker import features as F
from redrob_ranker import scoring
from redrob_ranker import reasoning
from redrob_ranker.io_utils import stream_candidates

KEEP = 200  # retain a margin above 100 so reasoning/tie handling has headroom


def run(candidates_path: str, out_path: str, weights=None, limit=None) -> None:
    weights = weights or scoring.DEFAULT_WEIGHTS
    t0 = time.time()
    heap = []  # min-heap of (score, base, candidate_id, candidate)
    counter = 0
    n = 0
    gmax = 1e-9  # global max internal score, for [0,1] display normalisation
    for cand in stream_candidates(candidates_path):
        n += 1
        if limit and n > limit:
            break
        cid = cand.get("candidate_id")
        if not cid:
            continue
        feats = F.extract(cand)
        info = scoring.score_candidate(cand, feats, weights)
        if info["score"] > gmax:
            gmax = info["score"]
        # tie-break key: prefer higher final, then higher base, then smaller id
        key = (info["score"], info["base"], cid)
        if len(heap) < KEEP:
            heapq.heappush(heap, (key, cand, feats, info))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, cand, feats, info))
        counter += 1

    # full ordering of the retained best
    best = sorted(heap, key=lambda x: x[0], reverse=True)

    # round to 4dp, then re-sort the top-100 by (-printed_score, candidate_id)
    # so the output strictly satisfies the validator's monotonicity + tie rules.
    rows = []
    for key, cand, feats, info in best[:120]:
        cid = cand.get("candidate_id")
        printed = round(min(1.0, info["score"] / gmax), 4)
        rows.append((printed, cid, cand, feats, info))
    rows.sort(key=lambda r: (-r[0], r[1]))
    top = rows[:100]

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (printed, cid, cand, feats, info) in enumerate(top, start=1):
            txt = reasoning.generate(cand, feats, info)
            w.writerow([cid, rank, f"{printed:.4f}", txt])

    dt = time.time() - t0
    hp_in_top = sum(1 for _, _, _, _, info in top if info["is_honeypot"])
    print(f"Scored {counter} candidates in {dt:.1f}s.")
    print(f"Wrote top 100 -> {out_path}")
    print(f"Honeypots in top 100: {hp_in_top} ({hp_in_top}%).")
    print(f"Top score {top[0][0]:.4f}  |  rank-100 score {top[-1][0]:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl[.gz]")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument("--weights", default=None, help="optional JSON file of tuned weights")
    ap.add_argument("--limit", type=int, default=None, help="process only first N (debug)")
    args = ap.parse_args(argv)

    weights = None
    if args.weights:
        with open(args.weights) as f:
            weights = json.load(f)
    run(args.candidates, args.out, weights=weights, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
