from __future__ import annotations
import pytest
from raggeneralization.evaluate import (
    span_f1,
    transfer_degradation,
    domain_adaptation_score,
    cross_domain_ndcg,
    technique_generalization_report,
    rank_by_generalization,
)
from raggeneralization.core import TransferResult, DomainAdaptationResult


# ---------------------------------------------------------------------------
# span_f1
# ---------------------------------------------------------------------------

def test_span_f1_perfect() -> None:
    spans = ["hello world"]
    assert span_f1(spans, spans) == pytest.approx(1.0)


def test_span_f1_no_overlap() -> None:
    assert span_f1(["foo"], ["bar"]) == pytest.approx(0.0)


def test_span_f1_empty_predicted() -> None:
    assert span_f1([], ["hello"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# transfer_degradation
# ---------------------------------------------------------------------------

def test_transfer_degradation_zero_source() -> None:
    assert transfer_degradation(0.0, 0.5) == pytest.approx(0.0)


def test_transfer_degradation_no_drop() -> None:
    assert transfer_degradation(0.8, 0.9) == pytest.approx(0.0)


def test_transfer_degradation_half() -> None:
    assert transfer_degradation(0.8, 0.4) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# domain_adaptation_score
# ---------------------------------------------------------------------------

def test_domain_adaptation_score_perfect() -> None:
    results = [
        DomainAdaptationResult(
            config_name="cfg_a",
            source_domain="legal-US",
            target_domain="legal-EU",
            source_ndcg=0.8,
            target_ndcg=0.8,
        )
    ]
    scores = domain_adaptation_score(results)
    assert scores["cfg_a"] == pytest.approx(1.0)


def test_domain_adaptation_score_zero() -> None:
    results = [
        DomainAdaptationResult(
            config_name="cfg_b",
            source_domain="legal-US",
            target_domain="legal-EU",
            source_ndcg=0.8,
            target_ndcg=0.0,
        )
    ]
    scores = domain_adaptation_score(results)
    assert scores["cfg_b"] == pytest.approx(0.0)


def test_domain_adaptation_score_multiple_configs() -> None:
    results = [
        DomainAdaptationResult("cfg_a", "legal-US", "legal-EU", 0.8, 0.64),
        DomainAdaptationResult("cfg_b", "legal-US", "legal-EU", 0.8, 0.72),
    ]
    scores = domain_adaptation_score(results)
    assert scores["cfg_a"] == pytest.approx(0.8)  # 0.64/0.8
    assert scores["cfg_b"] == pytest.approx(0.9)  # 0.72/0.8


def test_domain_adaptation_score_aggregates_multiple_pairs() -> None:
    """Mean adaptation score is computed when a config has multiple domain pairs."""
    results = [
        DomainAdaptationResult("cfg_a", "legal-US", "legal-EU", 1.0, 0.8),  # score=0.8
        DomainAdaptationResult("cfg_a", "legal-US", "medical", 1.0, 0.6),   # score=0.6
    ]
    scores = domain_adaptation_score(results)
    assert scores["cfg_a"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# cross_domain_ndcg
# ---------------------------------------------------------------------------

def test_cross_domain_ndcg_perfect() -> None:
    retrieved = {"q0": ["d0", "d1", "d2"]}
    qrels = {"q0": {"d0", "d1"}}
    scores = cross_domain_ndcg(retrieved, qrels, k=3)
    assert scores["q0"] == pytest.approx(1.0)
    assert scores["__mean__"] == pytest.approx(1.0)


def test_cross_domain_ndcg_no_hit() -> None:
    retrieved = {"q0": ["d9", "d8"]}
    qrels = {"q0": {"d0"}}
    scores = cross_domain_ndcg(retrieved, qrels, k=5)
    assert scores["q0"] == pytest.approx(0.0)


def test_cross_domain_ndcg_empty() -> None:
    scores = cross_domain_ndcg({}, {}, k=10)
    assert scores["__mean__"] == pytest.approx(0.0)


def test_cross_domain_ndcg_multiple_queries() -> None:
    retrieved = {
        "q0": ["d0", "d1"],  # perfect for q0
        "q1": ["d9", "d8"],  # no hit for q1
    }
    qrels = {"q0": {"d0"}, "q1": {"d0"}}
    scores = cross_domain_ndcg(retrieved, qrels, k=2)
    assert scores["q0"] == pytest.approx(1.0)
    assert scores["q1"] == pytest.approx(0.0)
    assert scores["__mean__"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# technique_generalization_report & rank_by_generalization
# ---------------------------------------------------------------------------

def _make_transfer_result(config: str, src_f1: float, tgt_f1: float) -> TransferResult:
    return TransferResult(
        source_domain="legal-US",
        target_domain="legal-EU",
        config_name=config,
        source_f1=src_f1,
        target_f1=tgt_f1,
    )


def test_generalization_report_keys() -> None:
    results = [
        _make_transfer_result("cfg_a", 0.8, 0.6),
        _make_transfer_result("cfg_b", 0.75, 0.7),
    ]
    report = technique_generalization_report(results)
    assert set(report.keys()) == {"cfg_a", "cfg_b"}
    assert "mean_transfer_gap" in report["cfg_a"]


def test_rank_by_generalization_order() -> None:
    results = [
        _make_transfer_result("worse", 0.8, 0.5),   # gap=0.3
        _make_transfer_result("better", 0.8, 0.75),  # gap=0.05
    ]
    ranked = rank_by_generalization(results)
    assert ranked[0]["config_name"] == "better"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["config_name"] == "worse"
