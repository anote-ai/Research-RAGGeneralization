"""Core data structures and benchmark runner for RAG generalization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ContractDocument:
    """A single legal contract document from CUAD."""

    contract_id: str
    text: str
    contract_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CUADQuestion:
    """A CUAD question with category label and reference answer spans."""

    question_id: str
    category: str
    question: str
    answer_spans: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for a retrieval-augmented generation pipeline."""

    chunking: str
    embedding: str
    reranker: str | None
    use_metadata: bool


# 5 representative CUAD question categories
CUAD_CATEGORIES: list[str] = [
    "Parties",
    "Effective Date",
    "Expiration Date",
    "Governing Law",
    "Termination for Convenience",
]


@dataclass
class TransferResult:
    """Records the domain transfer performance of a pipeline config."""

    source_domain: str
    target_domain: str
    config: PipelineConfig
    source_f1: float
    target_f1: float
    transfer_gap: float

    def __post_init__(self) -> None:
        expected_gap = self.source_f1 - self.target_f1
        # Allow a small floating-point tolerance
        if abs(self.transfer_gap - expected_gap) > 1e-9:
            raise ValueError(
                f"transfer_gap {self.transfer_gap} does not match "
                f"source_f1 - target_f1 = {expected_gap}"
            )


class GeneralizationBench:
    """Orchestrates CUAD loading and transfer evaluation."""

    def load_cuad_sample(self) -> list[ContractDocument]:
        """Load a sample of CUAD contracts.

        Stub implementation — override with real dataset loading.
        Returns an empty list by default.
        """
        return []

    def evaluate_transfer(
        self, config: PipelineConfig
    ) -> TransferResult:
        """Evaluate domain transfer for a given pipeline configuration.

        Stub implementation — override with actual pipeline execution.
        Returns a zero-gap TransferResult by default.
        """
        return TransferResult(
            source_domain="finance",
            target_domain="legal",
            config=config,
            source_f1=0.0,
            target_f1=0.0,
            transfer_gap=0.0,
        )
