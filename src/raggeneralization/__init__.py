"""raggeneralization: Domain transfer analysis of RAG pipelines from finance to legal (CUAD)."""

__version__ = "0.1.0"

from raggeneralization.core import (
    ContractDocument,
    CUAD_CATEGORIES,
    CUADQuestion,
    GeneralizationBench,
    PipelineConfig,
    TransferResult,
)

__all__ = [
    "ContractDocument",
    "CUADQuestion",
    "PipelineConfig",
    "CUAD_CATEGORIES",
    "TransferResult",
    "GeneralizationBench",
]
