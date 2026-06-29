"""Domain Gap Score (DGS): a real, runnable implementation of the metric
specified in DESIGN_DOC.md.

DESIGN_DOC.md defines:

    DGS = alpha * VocabDistance + beta * EmbeddingDistance
    VocabDistance = 1 - Jaccard(top_10K_terms(source), top_10K_terms(target))
    EmbeddingDistance = Wasserstein_distance(source_embeddings, target_embeddings)

This module implements ``vocab_distance`` exactly as specified. For
``EmbeddingDistance`` it implements a dependency-free proxy,
``token_distribution_distance``, using total-variation distance between
unigram frequency distributions over the shared vocabulary. This avoids
adding an embedding-model dependency (e.g. sentence-transformers) before
one is actually needed; swapping in a true embedding-based Wasserstein
distance is tracked as future work (see PAPER_DRAFT.md, Section 4.2).

Nothing in this module fabricates results: all functions operate on
whatever corpora are passed in and compute real statistics over them.
"""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _corpus_tokens(corpus: list[str]) -> list[str]:
    tokens: list[str] = []
    for doc in corpus:
        tokens.extend(_tokenize(doc))
    return tokens


def top_k_terms(corpus: list[str], top_k: int = 10_000) -> set[str]:
    """Return the *top_k* most frequent distinct terms in *corpus*.

    Args:
        corpus: List of raw text documents.
        top_k: Maximum number of distinct terms to keep.

    Returns:
        Set of the top_k most frequent lowercase alphanumeric tokens.
        If the corpus has fewer than top_k distinct terms, all of them
        are returned.
    """
    counts = Counter(_corpus_tokens(corpus))
    most_common = counts.most_common(top_k)
    return {term for term, _ in most_common}


def vocab_distance(
    corpus_a: list[str], corpus_b: list[str], top_k: int = 10_000
) -> float:
    """Vocabulary distance between two corpora, per DESIGN_DOC.md.

    ``VocabDistance = 1 - Jaccard(top_k_terms(a), top_k_terms(b))``

    Returns 0.0 when both corpora are empty (no information to compare,
    treated as no gap) and 1.0 when either corpus is empty but not both
    (maximal gap: completely disjoint vocabularies).
    """
    terms_a = top_k_terms(corpus_a, top_k)
    terms_b = top_k_terms(corpus_b, top_k)
    if not terms_a and not terms_b:
        return 0.0
    if not terms_a or not terms_b:
        return 1.0
    union = terms_a | terms_b
    intersection = terms_a & terms_b
    jaccard = len(intersection) / len(union) if union else 0.0
    return 1.0 - jaccard


def token_distribution_distance(corpus_a: list[str], corpus_b: list[str]) -> float:
    """Total-variation distance between unigram frequency distributions.

    This is a dependency-free proxy for DESIGN_DOC.md's
    ``EmbeddingDistance = Wasserstein_distance(source_embeddings, target_embeddings)``
    term. It captures shifts in token usage frequency (not just
    presence/absence, which ``vocab_distance`` already covers) without
    requiring an embedding model.

    Returns a value in [0, 1], where 0 means identical token-frequency
    distributions and 1 means completely disjoint support.
    """
    tokens_a = _corpus_tokens(corpus_a)
    tokens_b = _corpus_tokens(corpus_b)
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 1.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    vocab = set(counts_a) | set(counts_b)
    tv_distance = 0.0
    for term in vocab:
        p = counts_a.get(term, 0) / total_a
        q = counts_b.get(term, 0) / total_b
        tv_distance += abs(p - q)
    # Total variation distance is conventionally half the L1 distance,
    # which keeps the result bounded in [0, 1].
    return tv_distance / 2.0


def domain_gap_score(
    corpus_a: list[str],
    corpus_b: list[str],
    alpha: float = 0.5,
    beta: float = 0.5,
    top_k: int = 10_000,
) -> float:
    """Domain Gap Score (DGS) per DESIGN_DOC.md.

    ``DGS = alpha * VocabDistance + beta * EmbeddingDistance``

    Args:
        corpus_a: Source domain documents (raw text).
        corpus_b: Target domain documents (raw text).
        alpha: Weight on vocabulary distance. DESIGN_DOC.md specifies
            these weights should be "learned from training data" via the
            (not yet implemented) DGP model; until that exists, callers
            should treat the default 0.5/0.5 split as a placeholder, not
            a validated calibration.
        beta: Weight on the embedding/token-distribution distance term.
        top_k: Vocabulary cutoff passed to :func:`vocab_distance`.

    Returns:
        DGS value. With the default proxy distances, both components are
        bounded in [0, 1], so DGS is bounded in [0, alpha + beta].
    """
    v_dist = vocab_distance(corpus_a, corpus_b, top_k=top_k)
    e_dist = token_distribution_distance(corpus_a, corpus_b)
    return alpha * v_dist + beta * e_dist
