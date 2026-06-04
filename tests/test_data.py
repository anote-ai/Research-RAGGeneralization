from __future__ import annotations
from raggeneralization.data import (
    SAMPLE_CONTRACT_TEXTS,
    make_contract,
    make_cuad_question,
    make_transfer_results,
)
from raggeneralization.core import ContractDocument, CUADQuestion, TransferResult


def test_sample_contract_texts_non_empty() -> None:
    assert len(SAMPLE_CONTRACT_TEXTS) >= 1
    for text in SAMPLE_CONTRACT_TEXTS:
        assert len(text) > 50


def test_make_contract_type() -> None:
    doc = make_contract(0)
    assert isinstance(doc, ContractDocument)
    assert doc.contract_id == "contract_0000"


def test_make_cuad_question_type() -> None:
    q = make_cuad_question("Parties")
    assert isinstance(q, CUADQuestion)
    assert q.category == "Parties"
    assert q.has_answer


def test_make_transfer_results_length() -> None:
    results = make_transfer_results(n_configs=4)
    assert len(results) == 4
    for r in results:
        assert isinstance(r, TransferResult)
        assert 0.0 <= r.source_f1 <= 1.0
        assert 0.0 <= r.target_f1 <= 1.0
