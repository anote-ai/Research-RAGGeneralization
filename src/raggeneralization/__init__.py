"""RAGGeneralization: Domain transfer evaluation for RAG pipelines on contract data."""
from .core import (
    ContractDocument,
    CUADQuestion,
    PipelineConfig,
    CUAD_CATEGORIES,
    TransferResult,
    GeneralizationBench,
)
from .evaluate import (
    span_f1,
    transfer_degradation,
    technique_generalization_report,
    rank_by_generalization,
)

__all__ = [
    "ContractDocument",
    "CUADQuestion",
    "PipelineConfig",
    "CUAD_CATEGORIES",
    "TransferResult",
    "GeneralizationBench",
    "span_f1",
    "transfer_degradation",
    "technique_generalization_report",
    "rank_by_generalization",
]
__version__ = "0.1.0"
