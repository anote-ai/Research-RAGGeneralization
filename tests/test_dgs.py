from __future__ import annotations

import pytest

from raggeneralization.data import SAMPLE_CONTRACT_TEXTS
from raggeneralization.dgs import (
    domain_gap_score,
    token_distribution_distance,
    top_k_terms,
    vocab_distance,
)


# ---------------------------------------------------------------------------
# top_k_terms
# ---------------------------------------------------------------------------

def test_top_k_terms_basic() -> None:
    terms = top_k_terms(["the quick brown fox", "the lazy dog"], top_k=10)
    assert "the" in terms
    assert "fox" in terms


def test_top_k_terms_respects_limit() -> None:
    terms = top_k_terms(["a b c d e f g h"], top_k=3)
    assert len(terms) == 3


def test_top_k_terms_empty_corpus() -> None:
    assert top_k_terms([]) == set()


# ---------------------------------------------------------------------------
# vocab_distance
# ---------------------------------------------------------------------------

def test_vocab_distance_identical_corpora_is_zero() -> None:
    corpus = ["alpha beta gamma", "delta epsilon"]
    assert vocab_distance(corpus, corpus) == pytest.approx(0.0)


def test_vocab_distance_disjoint_corpora_is_one() -> None:
    corpus_a = ["alpha beta gamma"]
    corpus_b = ["delta epsilon zeta"]
    assert vocab_distance(corpus_a, corpus_b) == pytest.approx(1.0)


def test_vocab_distance_partial_overlap_between_zero_and_one() -> None:
    corpus_a = ["alpha beta gamma"]
    corpus_b = ["alpha delta epsilon"]
    d = vocab_distance(corpus_a, corpus_b)
    assert 0.0 < d < 1.0


def test_vocab_distance_both_empty_is_zero() -> None:
    assert vocab_distance([], []) == pytest.approx(0.0)


def test_vocab_distance_one_empty_is_one() -> None:
    assert vocab_distance(["alpha"], []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# token_distribution_distance
# ---------------------------------------------------------------------------

def test_token_distribution_distance_identical_is_zero() -> None:
    corpus = ["alpha alpha beta", "gamma gamma gamma delta"]
    assert token_distribution_distance(corpus, corpus) == pytest.approx(0.0)


def test_token_distribution_distance_disjoint_is_one() -> None:
    corpus_a = ["alpha alpha alpha"]
    corpus_b = ["beta beta beta"]
    assert token_distribution_distance(corpus_a, corpus_b) == pytest.approx(1.0)


def test_token_distribution_distance_bounded() -> None:
    corpus_a = ["alpha beta beta gamma"]
    corpus_b = ["alpha alpha beta gamma gamma gamma"]
    d = token_distribution_distance(corpus_a, corpus_b)
    assert 0.0 <= d <= 1.0


def test_token_distribution_distance_empty_corpora() -> None:
    assert token_distribution_distance([], []) == pytest.approx(0.0)
    assert token_distribution_distance(["alpha"], []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# domain_gap_score
# ---------------------------------------------------------------------------

def test_domain_gap_score_self_distance_is_zero() -> None:
    corpus = ["alpha beta gamma delta"]
    assert domain_gap_score(corpus, corpus) == pytest.approx(0.0)


def test_domain_gap_score_weights_sum_behavior() -> None:
    corpus_a = ["alpha beta gamma"]
    corpus_b = ["delta epsilon zeta"]
    # Fully disjoint vocab and token distributions -> both terms are 1.0
    score = domain_gap_score(corpus_a, corpus_b, alpha=0.5, beta=0.5)
    assert score == pytest.approx(1.0)


def test_domain_gap_score_respects_alpha_beta() -> None:
    corpus_a = ["alpha beta gamma"]
    corpus_b = ["delta epsilon zeta"]
    vocab_only = domain_gap_score(corpus_a, corpus_b, alpha=1.0, beta=0.0)
    assert vocab_only == pytest.approx(vocab_distance(corpus_a, corpus_b))


def test_domain_gap_score_on_synthetic_contracts() -> None:
    """(measured) DGS computed on the existing synthetic CUAD-style templates.

    This is a real measurement of the synthetic data already in
    src/raggeneralization/data.py -- NOT a finding about real US vs EU
    legal contract domain gap. The templates differ by contract type
    (NDA vs lease vs employment, etc.), not by genuine jurisdictional
    drafting style, so this number should not be cited as a research
    result. See PAPER_DRAFT.md Section 4.3 for the full caveat.
    """
    even = [t for i, t in enumerate(SAMPLE_CONTRACT_TEXTS) if i % 2 == 0]
    odd = [t for i, t in enumerate(SAMPLE_CONTRACT_TEXTS) if i % 2 == 1]
    score = domain_gap_score(even, odd)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # A corpus must have zero gap against itself.
    assert domain_gap_score(even, even) == pytest.approx(0.0)
    # The two distinct template groups should show *some* gap, since
    # they contain different contract types with different terminology.
    assert score > 0.0
