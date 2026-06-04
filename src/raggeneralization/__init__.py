"""RAGGeneralization: Domain transfer evaluation for RAG pipelines on contract data."""
from .core import (
    ContractDocument,
    CUADQuestion,
    PipelineConfig,
    CUAD_CATEGORIES,
    TransferResult,
    DomainAdaptationResult,
    GeneralizationBench,
    build_transfer_matrix,
)
from .evaluate import (
    span_f1,
    transfer_degradation,
    domain_adaptation_score,
    cross_domain_ndcg,
    technique_generalization_report,
    rank_by_generalization,
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
    "span_f1",
    "transfer_degradation",
    "domain_adaptation_score",
    "cross_domain_ndcg",
    "technique_generalization_report",
    "rank_by_generalization",
]
__version__ = "0.2.0"
