"""Rank the products that match a query by TF-IDF.

Documents and queries are weighted the same way: each term gets
(1 + log tf) * log(N / df), where tf is how often it appears in the document
(or query), df is how many documents contain it and N is the number of
documents. A product's score is the cosine similarity between its vector and
the query's, so long product texts are not favoured just for repeating words.
"""

import math

import pandas as pd

from fashion_search.index import build_index, match
from fashion_search.text import build_terms


class SearchEngine:
    """Search the product records: AND matching, then TF-IDF ranking."""

    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.set_index("pid", drop=False)
        self.index = build_index(corpus)
        n_docs = len(corpus)
        self.idf = {term: math.log(n_docs / len(posting)) for term, posting in self.index.items()}

        # Length of each product's TF-IDF vector, to normalise its scores.
        squares: dict[str, float] = {}
        for term, posting in self.index.items():
            for pid, tf in posting.items():
                squares[pid] = squares.get(pid, 0.0) + (_tf(tf) * self.idf[term]) ** 2
        self.norms = {pid: math.sqrt(total) for pid, total in squares.items()}

    def rank(self, query: str) -> list[tuple[str, float]]:
        """Every product matching the query with its score, best first."""
        query_terms = build_terms(query)
        matches = match(self.index, query_terms)
        if not matches:
            return []
        query_weights = {t: _tf(query_terms.count(t)) * self.idf[t] for t in set(query_terms)}
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

        scores = []
        for pid in matches:
            dot = sum(w * _tf(self.index[t][pid]) * self.idf[t] for t, w in query_weights.items())
            norm = self.norms[pid] * query_norm
            scores.append((pid, dot / norm if norm else 0.0))
        # Ties (e.g. identical products) are broken by pid, so results are reproducible.
        return sorted(scores, key=lambda s: (-s[1], s[0]))

    def search(self, query: str, k: int | None = None) -> list[str]:
        """Pids of the top k matching products (all of them if k is None), best first."""
        return [pid for pid, _ in self.rank(query)[:k]]

    def results(self, query: str, k: int = 10) -> pd.DataFrame:
        """The top k results with the fields a reader needs to judge them."""
        ranked = self.rank(query)[:k]
        table = self.corpus.loc[[pid for pid, _ in ranked], ["pid", "title", "brand", "sub_category"]]
        return table.assign(score=[score for _, score in ranked]).reset_index(drop=True)


def _tf(count: int) -> float:
    return 1 + math.log(count) if count else 0.0
