"""
Reward function for code ranking GRPO training.

The model outputs a ranking of candidate IDs (e.g., "2, 4, 1, 3").
Ground truth is a JSON list of pass_rates for each candidate.

Reward modes:
  - "pairacc": Pairwise accuracy — fraction of correctly ordered pairs.
  - "ndcg": Normalized Discounted Cumulative Gain.

Parse failure → reward = -1.

Set REWARD_MODE env var to switch (default: "pairacc").
"""

import json
import math
import os
import re


REWARD_MODE = os.environ.get("REWARD_MODE", "pairacc")


def parse_ranking(response: str, n_candidates: int) -> list[int] | None:
    """Parse ranking from model response.

    Returns list of 0-indexed candidate IDs in predicted order (best first),
    or None if parsing failed.
    """
    response = response.strip()

    # Strip thinking trace if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    numbers = re.findall(r'\d+', response)
    if not numbers:
        return None

    ranking = []
    seen = set()
    for num_str in numbers:
        if len(num_str) > 4:
            continue
        num = int(num_str)
        if 1 <= num <= n_candidates and num not in seen:
            ranking.append(num - 1)
            seen.add(num)

    if not ranking or len(ranking) != n_candidates:
        return None

    return ranking


def compute_ndcg(ranking: list[int], relevances: list[float], k: int | None = None) -> float:
    """Compute NDCG. Returns 1.0 if all relevances are equal."""
    if not ranking or not relevances:
        return 0.0

    if k is not None:
        ranking = ranking[:k]

    dcg = 0.0
    for i, cand_idx in enumerate(ranking):
        if 0 <= cand_idx < len(relevances):
            rel = relevances[cand_idx]
            dcg += (2 ** rel - 1) / math.log2(i + 2)

    sorted_rels = sorted(relevances, reverse=True)
    if k is not None:
        sorted_rels = sorted_rels[:k]

    idcg = 0.0
    for i, rel in enumerate(sorted_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 1.0

    return dcg / idcg


def compute_top1_acc(ranking: list[int], relevances: list[float]) -> float:
    """Whether the predicted top-1 has the highest pass_rate."""
    if not ranking or not relevances:
        return 0.0
    max_rel = max(relevances)
    if all(r == max_rel for r in relevances):
        return 1.0
    return 1.0 if relevances[ranking[0]] == max_rel else 0.0


def compute_pairacc(ranking: list[int], relevances: list[float]) -> float:
    """Fraction of correctly ordered pairs (skipping ties)."""
    if not ranking or not relevances:
        return 0.0

    pos = {cand_idx: rank_pos for rank_pos, cand_idx in enumerate(ranking)}

    correct = 0
    total = 0
    n = len(relevances)
    for i in range(n):
        for j in range(i + 1, n):
            if relevances[i] == relevances[j]:
                continue
            total += 1
            if relevances[i] > relevances[j] and pos.get(i, n) < pos.get(j, n):
                correct += 1
            elif relevances[i] < relevances[j] and pos.get(i, n) > pos.get(j, n):
                correct += 1

    if total == 0:
        return 1.0
    return correct / total


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    """verl-compatible reward function.

    Returns dict with: score (primary), ndcg, pairacc, top1_acc.
    """
    pass_rates = json.loads(ground_truth)
    n_candidates = len(pass_rates)

    ranking = parse_ranking(solution_str, n_candidates)

    if ranking is None:
        return {"score": -1.0, "ndcg": -1.0, "pairacc": -1.0, "top1_acc": -1.0}

    ndcg = compute_ndcg(ranking, pass_rates)
    pairacc = compute_pairacc(ranking, pass_rates)
    top1_acc = compute_top1_acc(ranking, pass_rates)
    score = ndcg if REWARD_MODE == "ndcg" else pairacc

    return {"score": score, "ndcg": ndcg, "pairacc": pairacc, "top1_acc": top1_acc}
