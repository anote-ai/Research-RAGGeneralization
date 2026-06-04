from __future__ import annotations
import pytest
from raggeneralization.evaluate import (
    span_f1,
    transfer_degradation,
    technique_generalization_report,
    rank_by_generalization,
)
from raggeneralization.core import TransferResult


def test_span_f1_exact() -> None:
    assert span_f1(["the cat sat"], ["the cat sat"]) == pytest.approx(1.0)


def test_span_f1_no_overlap() -> None:
    assert span_f1(["apple"], ["banana"]) == pytest.approx(0.0)


def test_span_f1_partial() -> None:
    score = span_f1(["the cat sat on the mat"], ["the cat ran on the floor"])
    assert 0.0 < score < 1.0


def test_span_f1_empty_predicted() -> None:
    assert span_f1([], ["hello world"]) == 0.0


def test_transfer_degradation_normal() -> None:
    assert transfer_degradation(0.80, 0.64) == pytest.approx(0.20)


def test_transfer_degradation_zero_source() -> None:
    assert transfer_degradation(0.0, 0.5) == 0.0


def test_technique_generalization_report_keys() -> None:
    results = [
        TransferResult("US", "EU", "cfg_a", 0.80, 0.65),
        TransferResult("US", "EU", "cfg_b", 0.75, 0.70),
    ]
    report = technique_generalization_report(results)
    assert "cfg_a" in report
    assert "cfg_b" in report
    for v in report.values():
        assert "mean_source_f1" in v
        assert "mean_target_f1" in v
        assert "mean_transfer_gap" in v
        assert "mean_relative_drop" in v


def test_rank_by_generalization_sorted() -> None:
    results = [
        TransferResult("US", "EU", "cfg_a", 0.80, 0.60),  # gap=0.20
        TransferResult("US", "EU", "cfg_b", 0.80, 0.72),  # gap=0.08
    ]
    ranked = rank_by_generalization(results)
    # Best generalizer (smallest gap) should be first
    assert ranked[0]["config_name"] == "cfg_b"
    assert ranked[1]["config_name"] == "cfg_a"


def test_report_single_result() -> None:
    results = [TransferResult("US", "EU", "only_cfg", 0.70, 0.55)]
    report = technique_generalization_report(results)
    assert "only_cfg" in report
    assert report["only_cfg"]["mean_transfer_gap"] == pytest.approx(0.15)
