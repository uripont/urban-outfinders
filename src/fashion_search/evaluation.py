"""Evaluation metrics for ranked search results.

Every metric takes `ranked`, the pids a search returned (best first), and
`labels`, the relevance judgements of one query as {pid: 0 or 1}. Products
without a judgement count as not relevant. Relevant products that the search
never returned still count as relevant, so they lower recall and AP.

`k` cuts the ranking to its top k results; for the metrics over several
queries (MAP, MRR), `k=None` uses each whole ranking.
"""

import math
from collections.abc import Iterable

import pandas as pd

# One query's evaluation input: (ranked pids, {pid: label}).
Run = tuple[list[str], dict[str, int]]


def _relevant(labels: dict[str, int]) -> int:
    """Number of products judged relevant for the query."""
    return sum(1 for label in labels.values() if label)


def precision_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Share of the top k results that are relevant (missing positions count as misses)."""
    return sum(labels.get(pid, 0) for pid in ranked[:k]) / k


def recall_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Share of all relevant products that appear in the top k results."""
    total = _relevant(labels)
    if total == 0:
        return 0.0
    return sum(labels.get(pid, 0) for pid in ranked[:k]) / total


def average_precision_at_k(ranked: list[str], labels: dict[str, int], k: int | None) -> float:
    """Mean of the precision at each rank (within the top k) that holds a relevant product.

    The sum is divided by the most relevant products the top k could hold,
    min(k, number relevant), so a perfect top k scores 1. With k=None the whole
    ranking is used and the sum is divided by the number of relevant products.
    """
    total = _relevant(labels)
    if total == 0:
        return 0.0
    top = ranked if k is None else ranked[:k]
    hits, precision_sum = 0, 0.0
    for rank, pid in enumerate(top, start=1):
        if labels.get(pid, 0):
            hits += 1
            precision_sum += hits / rank
    return precision_sum / (total if k is None else min(k, total))


def f1_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """Harmonic mean of P@k and R@k."""
    p = precision_at_k(ranked, labels, k)
    r = recall_at_k(ranked, labels, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def mean_average_precision(runs: Iterable[Run], k: int | None = None) -> float:
    """Mean of the average precision over several queries' (ranked, labels)."""
    aps = [average_precision_at_k(ranked, labels, k) for ranked, labels in runs]
    return sum(aps) / len(aps) if aps else 0.0


def mean_reciprocal_rank(runs: Iterable[Run], k: int | None = None) -> float:
    """Mean of 1 / rank of the first relevant result over several queries (0 if none in the top k)."""
    reciprocal_ranks = []
    for ranked, labels in runs:
        top = ranked if k is None else ranked[:k]
        first = next((rank for rank, pid in enumerate(top, start=1) if labels.get(pid, 0)), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    """DCG of the top k divided by the DCG of the best possible order of the judged products."""
    dcg = sum(labels.get(pid, 0) / math.log2(rank + 1) for rank, pid in enumerate(ranked[:k], start=1))
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate(runs: dict[int, Run], ks: Iterable[int] = (5, 10, 20)) -> pd.DataFrame:
    """P@K, R@K, AP@K, F1@K and NDCG@K for every query and k, rounded to three decimals."""
    rows = []
    for qid, (ranked, labels) in runs.items():
        for k in ks:
            rows.append({
                "query": qid,
                "K": k,
                "P@K": precision_at_k(ranked, labels, k),
                "R@K": recall_at_k(ranked, labels, k),
                "AP@K": average_precision_at_k(ranked, labels, k),
                "F1@K": f1_at_k(ranked, labels, k),
                "NDCG@K": ndcg_at_k(ranked, labels, k),
            })
    return pd.DataFrame(rows).set_index(["query", "K"]).round(3)


def load_labels(path) -> dict[int, dict[str, int]]:
    """Read a labels CSV (pid, query_id, labels) as {query_id: {pid: label}}."""
    df = pd.read_csv(path)
    return {
        qid: dict(zip(group.pid, group.labels))
        for qid, group in df.groupby("query_id")
    }
