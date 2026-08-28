"""Proof-step 1: does my closed-form |delta nDCG@5| match the OFFICIAL scorer?

Method: build random binary-labelled lists, rank them, then for every (pos, neg)
pair physically swap the two items, recompute nDCG@5 with kuairand-starter-kit's
own ndcg_at_k, and compare the empirical difference against the closed form.
Nothing is trusted to my own reimplementation of the metric.
"""
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/Users/khantminn/Desktop/Tiktok-TechJam-2026/kuairand-starter-kit")
from evaluate import ndcg_at_k  # the official implementation -- never reimplemented

K = 5


def inv_disc(pos: int) -> float:
    """1 / log2(pos+2), and exactly 0 beyond the @K cutoff (truncated away)."""
    return 1.0 / math.log2(pos + 2) if pos < K else 0.0


def idcg(labels) -> float:
    ideal = sorted(labels, reverse=True)[:K]
    return sum(((2 ** t) - 1) / math.log2(i + 2) for i, t in enumerate(ideal))


def closed_form_delta(labels_ranked, i, j) -> float:
    """|delta nDCG@5| from swapping ranks i and j. Standard LambdaRank weight."""
    ideal = idcg(labels_ranked)
    if ideal == 0:
        return 0.0
    gi = (2 ** labels_ranked[i]) - 1
    gj = (2 ** labels_ranked[j]) - 1
    return abs((gi - gj) * (inv_disc(i) - inv_disc(j))) / ideal


def empirical_delta(labels_ranked, i, j) -> float:
    """Physically swap and re-score with the OFFICIAL ndcg_at_k."""
    before = ndcg_at_k(labels_ranked, K)
    swapped = list(labels_ranked)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    after = ndcg_at_k(swapped, K)
    return abs(after - before)


def main():
    random.seed(0)
    checked = mismatches = 0
    worst = 0.0
    # cover the real distribution: 1..12 impressions spans median(4) through p90(12)
    for n in range(2, 13):
        for _trial in range(60):
            labels = [random.randint(0, 1) for _ in range(n)]
            if sum(labels) == 0 or sum(labels) == n:
                continue  # no (pos,neg) pair exists; no gradient under BPR either
            for i, j in itertools.combinations(range(n), 2):
                if labels[i] == labels[j]:
                    continue  # LambdaRank only weights pos-vs-neg pairs
                cf = closed_form_delta(labels, i, j)
                em = empirical_delta(labels, i, j)
                checked += 1
                diff = abs(cf - em)
                worst = max(worst, diff)
                if diff > 1e-12:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  MISMATCH n={n} labels={labels} swap({i},{j}): "
                              f"closed={cf:.10f} empirical={em:.10f}")
    print(f"pairs checked: {checked}")
    print(f"mismatches   : {mismatches}")
    print(f"max abs diff : {worst:.3e}")
    print("RESULT:", "PASS -- closed form matches the official scorer exactly"
          if mismatches == 0 else "FAIL")


if __name__ == "__main__":
    main()
