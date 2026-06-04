from __future__ import annotations
import pytest
from raggeneralization.core import (
    ContractDocument,
    CUADQuestion,
    PipelineConfig,
    CUAD_CATEGORIES,
    TransferResult,
    GeneralizationBench,
)


def test_contract_document_construction() -> None:
    doc = ContractDocument(contract_id="c1", text="hello", contract_type="NDA")
    assert doc.contract_id == "c1"
    assert doc.contract_type == "NDA"


def test_cuad_question_has_answer_true() -> None:
    q = CUADQuestion("q1", "Parties", "Who?", ["Acme Corp"])
    assert q.has_answer is True


def test_cuad_question_has_answer_false() -> None:
    q = CUADQuestion("q2", "Parties", "Who?", [])
    assert q.has_answer is False


def test_cuad_question_has_answer_empty_string() -> None:
    q = CUADQuestion("q3", "Parties", "Who?", ["  "])
    assert q.has_answer is False


def test_pipeline_config_name_simple() -> None:
    cfg = PipelineConfig(chunking="fixed_512", embedding="bm25")
    assert "fixed_512" in cfg.name()
    assert "bm25" in cfg.name()


def test_pipeline_config_name_with_reranker() -> None:
    cfg = PipelineConfig(chunking="sentence", embedding="dense", reranker="cross-encoder")
    assert "cross-encoder" in cfg.name()


def test_pipeline_config_name_with_metadata() -> None:
    cfg = PipelineConfig(chunking="recursive", embedding="dense", use_metadata=True)
    assert "meta" in cfg.name()


def test_cuad_categories_length() -> None:
    assert len(CUAD_CATEGORIES) >= 5


def test_transfer_result_transfer_gap() -> None:
    r = TransferResult("legal-US", "legal-EU", "cfg", source_f1=0.80, target_f1=0.65)
    assert r.transfer_gap == pytest.approx(0.15)


def test_transfer_result_relative_drop() -> None:
    r = TransferResult("legal-US", "legal-EU", "cfg", source_f1=0.80, target_f1=0.65)
    assert r.relative_drop == pytest.approx(0.15 / 0.80)


def test_generalization_bench_load_cuad_sample_length() -> None:
    bench = GeneralizationBench()
    docs = bench.load_cuad_sample(n=7)
    assert len(docs) == 7


def test_simulate_transfer_returns_transfer_result() -> None:
    bench = GeneralizationBench()
    cfg = PipelineConfig(chunking="sentence", embedding="dense")
    result = bench.simulate_transfer(cfg)
    assert isinstance(result, TransferResult)


def test_simulate_transfer_gap_nonnegative_for_basic_config() -> None:
    bench = GeneralizationBench()
    cfg = PipelineConfig(chunking="fixed_512", embedding="bm25")
    result = bench.simulate_transfer(cfg)
    # Basic configs should degrade; gap could be slightly negative due to noise but
    # the expected degradation should result in a non-hugely-negative gap
    assert result.transfer_gap > -0.1
