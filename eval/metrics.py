"""
Ranking metrics matching the challenge's scoring formula:

    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

All functions take a list of graded relevance labels in *predicted* rank order
(index 0 = your rank 1).  "Relevant" for binary metrics defaults to tier >= 3,
matching the spec's "P@10 = fraction of top-10 that are relevant (tier 3+)".
"""
from __future__ import annotations
import math
from typing import List

REL_THRESHOLD = 3


def dcg_at_k(rels: List[float], k: int) -> float:
    return sum((2 ** rels[i] - 1) / math.log2(i + 2) for i in range(min(k, len(rels))))


def ndcg_at_k(rels: List[float], k: int) -> float:
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def average_precision(rels: List[float], threshold: int = REL_THRESHOLD) -> float:
    hits, ap = 0, 0.0
    n_rel = sum(1 for r in rels if r >= threshold)
    if n_rel == 0:
        return 0.0
    for i, r in enumerate(rels):
        if r >= threshold:
            hits += 1
            ap += hits / (i + 1)
    return ap / n_rel


def precision_at_k(rels: List[float], k: int, threshold: int = REL_THRESHOLD) -> float:
    top = rels[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r >= threshold) / len(top)


def composite(rels: List[float]) -> dict:
    out = {
        "ndcg@10": ndcg_at_k(rels, 10),
        "ndcg@50": ndcg_at_k(rels, 50),
        "map": average_precision(rels),
        "p@10": precision_at_k(rels, 10),
    }
    out["composite"] = (0.50 * out["ndcg@10"] + 0.30 * out["ndcg@50"]
                        + 0.15 * out["map"] + 0.05 * out["p@10"])
    return out
