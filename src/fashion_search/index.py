"""Inverted index over the product records, with conjunctive (AND) matching.

The index maps each term to the products whose searchable text contains it
(title, description, product details and brand, see `corpus.all_terms`),
with the number of times it appears in each, which TF-IDF ranking needs.
A query matches the products that contain every one of its terms.
"""

from collections import Counter, defaultdict

import pandas as pd

from fashion_search.corpus import all_terms

# term -> {pid: number of times the term appears in that product}
Index = dict[str, dict[str, int]]


def build_index(corpus: pd.DataFrame) -> Index:
    """Map each term to the products that contain it and how often."""
    index: dict[str, dict[str, int]] = defaultdict(dict)
    for _, record in corpus.iterrows():
        for term, count in Counter(all_terms(record)).items():
            index[term][record["pid"]] = count
    return dict(index)


def match(index: Index, query_terms: list[str]) -> set[str]:
    """Pids of the products that contain every query term (AND)."""
    if not query_terms:
        return set()
    # Intersect from the shortest posting list, so each step only shrinks the set.
    postings = sorted((index.get(term, {}) for term in set(query_terms)), key=len)
    matches = set(postings[0])
    for posting in postings[1:]:
        matches.intersection_update(posting)
    return matches
