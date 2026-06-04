"""Tests for raggeneralization.core."""

import pytest

from raggeneralization.core import (
    ContractDocument,
    CUAD_CATEGORIES,
    CUADQuestion,
    GeneralizationBench,
    PipelineConfig,
    TransferResult,
)


# --- ContractDocument ---

def test_contract_document_construction():
    doc = ContractDocument(
        contract_id="cuad_001",
        text="This agreement is entered into by Acme Corp and Beta LLC...",
        contract_type="NDA",
        metadata={"pages": 12},
    )
    assert doc.contract_id == "cuad_001"
    assert doc.contract_type == "NDA"
    assert doc.metadata["pages"] == 12


def test_contract_document_default_metadata():
    doc = ContractDocument(contract_id="c1", text="text", contract_type="SaaS")
    assert doc.metadata == {}


# --- CUADQuestion ---

def test_cuad_question_construction():
    q = CUADQuestion(
        question_id="q42",
        category="Governing Law",
        question="What jurisdiction governs this agreement?",
        answer_spans=["State of Delaware"],
    )
    assert q.question_id == "q42"
    assert q.category == "Governing Law"
    assert len(q.answer_spans) == 1


def test_cuad_question_default_spans():
    q = CUADQuestion(question_id="q1", category="Parties", question="Who are the parties?")
    assert q.answer_spans == []


# --- PipelineConfig ---

def test_pipeline_config_construction():
    cfg = PipelineConfig(
        chunking="semantic",
        embedding="text-embedding-3-large",
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_metadata=True,
    )
    assert cfg.chunking == "semantic"
    assert cfg.reranker is not None


def test_pipeline_config_no_reranker():
    cfg = PipelineConfig(
        chunking="fixed_512",
        embedding="text-embedding-ada-002",
        reranker=None,
        use_metadata=False,
    )
    assert cfg.reranker is None


# --- CUAD_CATEGORIES ---

def test_cuad_categories_length():
    assert len(CUAD_CATEGORIES) == 5


def test_cuad_categories_content():
    assert "Parties" in CUAD_CATEGORIES
    assert "Governing Law" in CUAD_CATEGORIES


# --- TransferResult ---

def test_transfer_result_construction():
    cfg = PipelineConfig(chunking="sentence", embedding="ada", reranker=None, use_metadata=True)
    tr = TransferResult(
        source_domain="finance",
        target_domain="legal",
        config=cfg,
        source_f1=0.80,
        target_f1=0.65,
        transfer_gap=0.15,
    )
    assert tr.transfer_gap == pytest.approx(0.15)


def test_transfer_result_gap_mismatch_raises():
    cfg = PipelineConfig(chunking="fixed_512", embedding="ada", reranker=None, use_metadata=False)
    with pytest.raises(ValueError, match="transfer_gap"):
        TransferResult(
            source_domain="finance",
            target_domain="legal",
            config=cfg,
            source_f1=0.80,
            target_f1=0.65,
            transfer_gap=0.20,  # incorrect
        )


# --- GeneralizationBench stubs ---

def test_generalization_bench_load_stub():
    bench = GeneralizationBench()
    docs = bench.load_cuad_sample()
    assert isinstance(docs, list)


def test_generalization_bench_evaluate_stub():
    bench = GeneralizationBench()
    cfg = PipelineConfig(chunking="fixed_512", embedding="ada", reranker=None, use_metadata=False)
    result = bench.evaluate_transfer(cfg)
    assert isinstance(result, TransferResult)
    assert result.source_domain == "finance"
