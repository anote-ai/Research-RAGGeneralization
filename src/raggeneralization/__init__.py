"""raggeneralization: Domain transfer analysis of RAG pipelines from finance to legal (CUAD)."""

__version__ = "0.1.0"

from raggeneralization.core import (
    ContractDocument,
    CUAD_CATEGORIES,
    CUADQuestion,
    DomainAdaptationResult,
    GeneralizationBench,
    PipelineConfig,
    TransferResult,
    build_transfer_matrix,
)
from raggeneralization.dgs import (
    domain_gap_score,
    token_distribution_distance,
    top_k_terms,
    vocab_distance,
)

__all__ = [
    "ContractDocument",
    "CUADQuestion",
    "PipelineConfig",
    "CUAD_CATEGORIES",
    "TransferResult",
    "DomainAdaptationResult",
    "GeneralizationBench",
    "build_transfer_matrix",
    "domain_gap_score",
    "vocab_distance",
    "token_distribution_distance",
    "top_k_terms",
]
