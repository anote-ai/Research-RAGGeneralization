from __future__ import annotations
import pytest
from raggeneralization.core import (
    ContractDocument,
    CUADQuestion,
    PipelineConfig,
    TransferResult,
    DomainAdaptationResult,
    GeneralizationBench,
    build_transfer_matrix,
    CUAD_CATEGORIES,
)


# ---------------------------------------------------------------------------
# DomainAdaptationResult
# ---------------------------------------------------------------------------

def test_adaptation_score_perfect() -> None:
    r = DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.8, 0.8)
    assert r.adaptation_score == pytest.approx(1.0)


def test_adaptation_score_clamped() -> None:
    """target_ndcg > source_ndcg should still give score <= 1.0."""
    r = DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.5, 0.9)
    assert r.adaptation_score == pytest.approx(1.0)


def test_adaptation_score_zero_source() -> None:
    r = DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.0, 0.5)
    assert r.adaptation_score == pytest.approx(0.0)


def test_ndcg_gap() -> None:
    r = DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.8, 0.6)
    assert r.ndcg_gap == pytest.approx(0.2)


def test_ndcg_gap_no_drop() -> None:
    r = DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.6, 0.8)
    assert r.ndcg_gap == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_transfer_matrix
# ---------------------------------------------------------------------------

def test_build_transfer_matrix_diagonal() -> None:
    results = [
        DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.8, 0.7),
        DomainAdaptationResult("cfg", "legal-EU", "legal-US", 0.75, 0.65),
    ]
    matrix = build_transfer_matrix(results)
    assert matrix["legal-US"]["legal-US"] == pytest.approx(1.0)
    assert matrix["legal-EU"]["legal-EU"] == pytest.approx(1.0)


def test_build_transfer_matrix_values() -> None:
    results = [
        DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.8, 0.8),  # score=1.0
        DomainAdaptationResult("cfg", "legal-US", "legal-EU", 0.8, 0.4),  # score=0.5
    ]
    matrix = build_transfer_matrix(results)
    # Mean of 1.0 and 0.5
    assert matrix["legal-US"]["legal-EU"] == pytest.approx(0.75)


def test_build_transfer_matrix_empty() -> None:
    matrix = build_transfer_matrix([])
    assert matrix == {}


# ---------------------------------------------------------------------------
# GeneralizationBench.simulate_adaptation
# ---------------------------------------------------------------------------

def test_simulate_adaptation_returns_result() -> None:
    bench = GeneralizationBench()
    config = PipelineConfig(chunking="semantic", embedding="dense", reranker="cross-encoder")
    result = bench.simulate_adaptation(config)
    assert isinstance(result, DomainAdaptationResult)
    assert 0.0 <= result.source_ndcg <= 1.0
    assert 0.0 <= result.target_ndcg <= 1.0


def test_simulate_adaptation_score_between_zero_and_one() -> None:
    bench = GeneralizationBench()
    config = PipelineConfig(chunking="fixed_512", embedding="bm25")
    result = bench.simulate_adaptation(config)
    assert 0.0 <= result.adaptation_score <= 1.0


# ---------------------------------------------------------------------------
# CUAD_CATEGORIES expanded
# ---------------------------------------------------------------------------

def test_cuad_categories_extended() -> None:
    assert "Non-Compete" in CUAD_CATEGORIES
    assert "Confidentiality" in CUAD_CATEGORIES
    assert "Dispute Resolution" in CUAD_CATEGORIES
    assert "Indemnification" in CUAD_CATEGORIES


# ---------------------------------------------------------------------------
# Existing tests (retained)
# ---------------------------------------------------------------------------

def test_transfer_result_transfer_gap() -> None:
    r = TransferResult("legal-US", "legal-EU", "cfg", 0.8, 0.65)
    assert r.transfer_gap == pytest.approx(0.15)


def test_transfer_result_relative_drop() -> None:
    r = TransferResult("legal-US", "legal-EU", "cfg", 0.8, 0.6)
    assert r.relative_drop == pytest.approx(0.25)


def test_pipeline_config_name_with_reranker() -> None:
    cfg = PipelineConfig(chunking="sentence", embedding="dense", reranker="cross-encoder")
    assert "cross-encoder" in cfg.name()


def test_generalization_bench_load_cuad_sample() -> None:
    bench = GeneralizationBench()
    docs = bench.load_cuad_sample(n=7)
    assert len(docs) == 7
    assert all(isinstance(d, ContractDocument) for d in docs)


def test_cuad_question_has_answer() -> None:
    q = CUADQuestion("q1", "Parties", "Who are the parties?", ["Acme", "Beta"])
    assert q.has_answer is True


def test_cuad_question_no_answer() -> None:
    q = CUADQuestion("q2", "Parties", "Who are the parties?", [])
    assert q.has_answer is False
