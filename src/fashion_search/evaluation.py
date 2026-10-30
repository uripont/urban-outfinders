"""Evaluation metrics for ranked search results.

Every metric takes `ranked`, the pids a search returned (best first), and
`labels`, the relevance judgements of one query as {pid: 0 or 1}. Products
without a judgement count as not relevant.
"""

import pandas as pd


def precision_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    raise NotImplementedError


def recall_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    raise NotImplementedError


def average_precision_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    raise NotImplementedError


def f1_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    raise NotImplementedError


def mean_average_precision(runs: list[tuple[list[str], dict[str, int]]], k: int) -> float:
    """Mean of the average precision over several queries' (ranked, labels)."""
    raise NotImplementedError


def mean_reciprocal_rank(runs: list[tuple[list[str], dict[str, int]]], k: int) -> float:
    """Mean of 1 / rank of the first relevant result over several queries."""
    raise NotImplementedError


def ndcg_at_k(ranked: list[str], labels: dict[str, int], k: int) -> float:
    raise NotImplementedError


def load_labels(path) -> dict[int, dict[str, int]]:
    """Read a labels CSV (pid, query_id, labels) as {query_id: {pid: label}}."""
    df = pd.read_csv(path)
    return {
        qid: dict(zip(group.pid, group.labels))
        for qid, group in df.groupby("query_id")
    }
