"""Tests for raggeneralization.evaluate."""

import pytest

from raggeneralization.core import PipelineConfig, TransferResult
from raggeneralization.evaluate import (
    span_f1,
    technique_generalization_report,
    transfer_degradation,
)


def _make_cfg(chunking: str = "fixed_512", reranker: str | None = None) -> PipelineConfig:
    return PipelineConfig(
        chunking=chunking,
        embedding="ada",
        reranker=reranker,
        use_metadata=False,
    )


def _make_result(source_f1: float, target_f1: float, **cfg_kwargs) -> TransferResult:
    cfg = _make_cfg(**cfg_kwargs)
    gap = source_f1 - target_f1
    return TransferResult(
        source_domain="finance",
        target_domain="legal",
        config=cfg,
        source_f1=source_f1,
        target_f1=target_f1,
        transfer_gap=gap,
    )


# --- span_f1 ---

def test_span_f1_perfect():
    assert span_f1(["State of Delaware"], ["State of Delaware"]) == pytest.approx(1.0)


def test_span_f1_partial_overlap():
    score = span_f1(["Delaware law"], ["governed by Delaware"])
    assert 0.0 < score < 1.0


def test_span_f1_no_overlap():
    assert span_f1(["apple orange"], ["banana grape"]) == pytest.approx(0.0)


def test_span_f1_both_empty():
    assert span_f1([], []) == pytest.approx(1.0)


def test_span_f1_one_empty():
    assert span_f1([], ["Delaware"]) == pytest.approx(0.0)


# --- transfer_degradation ---

def test_transfer_degradation_basic():
    result = transfer_degradation(source_f1=0.80, target_f1=0.64)
    assert result == pytest.approx(0.20)


def test_transfer_degradation_no_gap():
    assert transfer_degradation(0.70, 0.70) == pytest.approx(0.0)


def test_transfer_degradation_zero_source():
    assert transfer_degradation(0.0, 0.5) == 0.0


# --- technique_generalization_report ---

def test_generalization_report_groups_by_config():
    results = [
        _make_result(0.80, 0.65, chunking="semantic"),
        _make_result(0.80, 0.55, chunking="semantic"),
        _make_result(0.75, 0.70, chunking="fixed_512"),
    ]
    report = technique_generalization_report(results)
    assert len(report) == 2
    semantic_key = next(k for k in report if "chunking=semantic" in k)
    assert report[semantic_key]["mean_transfer_gap"] == pytest.approx(0.20)


def test_generalization_report_mean_f1():
    results = [
        _make_result(0.80, 0.70, chunking="sentence"),
        _make_result(0.60, 0.50, chunking="sentence"),
    ]
    report = technique_generalization_report(results)
    key = next(k for k in report if "chunking=sentence" in k)
    assert report[key]["mean_source_f1"] == pytest.approx(0.70)
    assert report[key]["mean_target_f1"] == pytest.approx(0.60)
