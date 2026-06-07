#!/usr/bin/env python3
"""
Redrob ranker — entrypoint.

    python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline:
  1. Split the JSONL by byte offsets, one chunk per CPU core (Python stdlib
     `multiprocessing`).
  2. In each worker, stream candidates and skip the obvious non-fits with a
     cheap pre-filter (no AI/ML title, no AI skills, no IR/ML hint in the
     headline) — this cuts about 75% of the 100K pool before doing the
     expensive ~150-phrase text scan.
  3. The surviving candidates get the full feature extraction, scoring
     (7-component weighted sum modified by disqualifier / behavioural /
     honeypot multipliers), and a local heap of the top KEEP candidates.
  4. Worker heaps are merged in the main process and the top-100 is written.

Compute profile: CPU only, no network, stdlib only.  On an 8-core M-series box,
runs the full 100K pool in ~3-4 s and stays well under the Stage-3 limits
(~63 MB aggregate RSS, ~8 MB per worker).
"""
from __future__ import annotations
import argparse
import csv
import heapq
import json
import multiprocessing as mp
import os
import sys
import time

from redrob_ranker import features as F
from redrob_ranker import scoring
from redrob_ranker import reasoning
from redrob_ranker import prefilter

KEEP = 200  # retain a margin above 100 so reasoning/tie handling has headroom


def _chunk_offsets(path: str, n: int):
    """Split a file into ~equal byte ranges. Workers seek to start, drop the
    partial first line, and read until tell() >= end."""
    size = os.path.getsize(path)
    step = max(1, size // n)
    return [(i * step, ((i + 1) * step) if i < n - 1 else size) for i in range(n)]


def _worker(args):
    """One worker: scan a byte range, return (heap, gmax, scanned, prefiltered)."""
    path, start, end, keep, weights = args
    heap = []
    gmax = 1e-9
    scanned = 0
    prefiltered = 0
    with open(path, "r", encoding="utf-8") as fh:
        if start > 0:
            fh.seek(start)
            fh.readline()  # drop partial line
        while True:
            pos = fh.tell()
            if pos >= end:
                break
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = cand.get("candidate_id")
            if not cid:
                continue
            scanned += 1
            if not prefilter.cheap_can_compete(cand):
                prefiltered += 1
                continue
            feats = F.extract(cand)
            info = scoring.score_candidate(cand, feats, weights)
            if info["score"] > gmax:
                gmax = info["score"]
            key = (info["score"], info["base"], cid)
            if len(heap) < keep:
                heapq.heappush(heap, (key, cand, feats, info))
            elif key > heap[0][0]:
                heapq.heapreplace(heap, (key, cand, feats, info))
    return heap, gmax, scanned, prefiltered


def run(candidates_path: str, out_path: str, weights=None, n_workers=None) -> None:
    weights = weights or scoring.DEFAULT_WEIGHTS
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4))
    t0 = time.time()

    chunks = _chunk_offsets(candidates_path, n_workers)
    args_list = [(candidates_path, s, e, KEEP, weights) for s, e in chunks]

    if n_workers == 1:
        results = [_worker(args_list[0])]
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            results = pool.map(_worker, args_list)

    total_scanned = sum(r[2] for r in results)
    total_prefiltered = sum(r[3] for r in results)
    gmax = max(r[1] for r in results)

    merged = []
    for heap, _, _, _ in results:
        merged.extend(heap)
    merged.sort(key=lambda x: x[0], reverse=True)
    best = merged[:120]

    rows = []
    for key, cand, feats, info in best:
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
    full_scored = total_scanned - total_prefiltered
    pre_pct = 100.0 * total_prefiltered / max(1, total_scanned)
    print(f"Scored {total_scanned} candidates in {dt:.1f}s using {n_workers} workers.")
    print(f"  pre-filtered out: {total_prefiltered} ({pre_pct:.1f}%) — full feature scan ran on {full_scored}")
    print(f"Wrote top 100 -> {out_path}")
    print(f"Honeypots in top 100: {hp_in_top} ({hp_in_top}%).")
    print(f"Top score {top[0][0]:.4f}  |  rank-100 score {top[-1][0]:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl[.gz]")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument("--weights", default=None, help="optional JSON file of tuned weights")
    ap.add_argument("--workers", type=int, default=None, help="worker count (default: cpu_count)")
    args = ap.parse_args(argv)

    weights = None
    if args.weights:
        with open(args.weights) as f:
            weights = json.load(f)
    run(args.candidates, args.out, weights=weights, n_workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
